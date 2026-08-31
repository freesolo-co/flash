"""Allocation-free tests for the hosted per-model capacity benchmark.

Two jobs. First, prove the benchmark cannot touch production: not the app name, not the catalog, not
the router, not the billing path. Second, prove the metric arithmetic is right against hand-computed
values, so a number in the published report is trustworthy without re-deriving it.

Everything here runs with no GPU and no network.
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import contextlib
import dataclasses
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import tomllib
import types
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
    _POOL_PERIOD_SLACK,
    REQUEST_TIMEOUT_SECONDS,
    _absorb_event,
    _build_prompt_pool,
    _drain,
    _prompt_issuer,
    _StreamOutcome,
    _validate,
    base_model_record,
    drain_reap_seconds,
    fitting_watchdog_grace_seconds,
    prompt_fit_seconds_bound,
    run_cell,
    run_request_bound_seconds,
)
from flash.serving.bench.metrics import (
    ERROR_CACHE_CONTAMINATED,
    ERROR_CACHE_UNVERIFIED,
    ERROR_ENGINE,
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
from flash.serving.bench.warmup import (
    CANARY_WARMUP_REQUESTS,
    run_warmup,
    warmup_fit_seconds_bound,
)
from flash.serving.bench.workload import (
    BUCKETS,
    BUCKETS_BY_NAME,
    ENABLE_THINKING,
    N_CHOICES,
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
    """``@app.cls(gpu=...)`` fixes the GPU at decoration time, so each tier needs its own class.

    Importing the module would pull in ``modal`` and the whole serving stack, so the structure is
    asserted from the source: the factory must be called once per DISTINCT tier in the catalog.

    The catalog's tier COUNT is dev's to choose, not this harness's. It has been three (one card
    per model) and is currently one (#1376 moved every hosted model to B200). Asserting a minimum
    of two pinned the harness to a catalog shape dev had already abandoned, so the invariant is
    keyed to the distinct tiers themselves: whatever dev assigns, the app builds exactly that set.
    """
    source = BENCH_APP.read_text(encoding="utf-8")
    assert "_build_bench_engine" in source
    assert "ENGINE_BY_GPU" in source
    # No single hardcoded GPU: the tier comes from the catalog, per model.
    assert "GPU_SPEC" not in source
    tiers = {bench_gpu_for(model) for model in BENCH_MODELS}
    assert tiers, "the catalog resolved no GPU tier at all"
    # One class per distinct tier -- never one class spanning two, and never a stale extra.
    built = set(re.findall(r"_build_bench_engine\(\s*[\"\']([A-Za-z0-9_.-]+)[\"\']", source))
    if built:
        assert built == tiers, f"app builds classes for {built}, catalog assigns {tiers}"
    else:
        # Built by comprehension over the catalog's own distinct tiers, which is stronger.
        assert "_distinct_bench_gpus()" in source


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
    """Presence of ``_resolve_gdn_prefill_backend`` says nothing about which backend it picks.

    The call lives in `_gdn_backend_in_process` because the resolver reaches
    `torch.cuda.get_device_capability` and would create a CUDA context; see the isolation test
    below. It is still the resolver's own decision being recorded, not a reproduction of its rule.
    """
    from flash.serving.bench.probe import _gdn_backend_in_process

    source = inspect.getsource(_gdn_backend_in_process)
    assert "resolver(**kwargs)" in source, "the resolver must be CALLED, not merely found"
    call_at = source.index("resolver(**kwargs)")
    present_at = source.index('result["resolver_present"]')
    assert call_at > present_at


def test_the_gdn_resolver_never_runs_in_the_engine_parent_process() -> None:
    """The resolver must run in a throwaway child, never beside the engine it certifies.

    `_resolve_gdn_prefill_backend` queries `current_platform.get_device_capability`, whose CUDA
    implementation calls `torch.cuda.get_device_capability`. Running that here would create a CUDA
    context in the long-lived Modal parent -- defeating `probe_gpu`'s deliberate NVML-only design
    and consuming the post-init headroom whose absence OOM-killed the 35B engine's first-request
    workspace. The probe would then break the engine it exists to certify.

    Deliberately NOT re-derived from the NVML capability already collected: that would be a second
    implementation of a vLLM-internal rule, free to drift from the one the engine actually runs.
    """
    from flash.serving.bench import probe as probe_module

    wrapper = inspect.getsource(probe_module.probe_gdn_backend)
    assert "subprocess.run" in wrapper, "the resolver is not isolated in a child process"
    assert "_gdn_backend_in_process" in wrapper, "the child does not run the real resolver"
    # No shell, and the model name travels as data rather than in the command.
    assert "shell=True" not in wrapper
    assert "sys.executable" in wrapper

    # Every failure mode degrades to an unresolved probe; a gate must never see an invented backend.
    for stub in (
        mock.Mock(side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1.0)),
        mock.Mock(side_effect=OSError("no interpreter")),
        mock.Mock(return_value=types.SimpleNamespace(returncode=1, stdout="", stderr="boom")),
        mock.Mock(return_value=types.SimpleNamespace(returncode=0, stdout="not json", stderr="")),
    ):
        with mock.patch("flash.serving.bench.probe.subprocess.run", stub):
            result = probe_module.probe_gdn_backend("Qwen/Qwen3.5-9B")
        assert result["resolved"] is None, "a failed subprocess produced a backend anyway"
        assert result.get("reason"), "an unresolved probe must say why"
        assert result["subprocess_isolated"] is True


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
    # `_make_spawner` builds the pool through `_prompt_issuer`, which fits every prompt eagerly,
    # and only then starts the clock it returns -- so the ordering assertion is against the spawner.
    # `run_cell` takes `origin` FROM it rather than starting its own, which is what keeps the two
    # in that order; a `run_cell` that stamped its own origin could open the window before fitting.
    from flash.serving.bench.driver import _make_spawner

    source = inspect.getsource(_make_spawner)
    pool_at = source.index("_prompt_issuer(")
    origin_at = source.index("origin = time.monotonic()")
    assert pool_at < origin_at, "prompt pool must be built before the measured window opens"
    assert "fit_prompt_to_tokens" not in source[origin_at:], "fitting leaked into the measured loop"

    cell = inspect.getsource(run_cell)
    assert "origin, _spawn" in cell, "run_cell does not take its clock from the fitted pool"
    assert "origin = time.monotonic()" not in cell, (
        "run_cell stamps its own origin, so the window can open before the pool is fitted"
    )
    assert "fit_prompt_to_tokens" not in cell, "fitting leaked into the measured loop"

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
    from flash.serving.bench.driver import _make_spawner

    assert "_prompt_issuer(" in inspect.getsource(_make_spawner), (
        "the spawner run_cell uses bypasses the issuer"
    )
    assert "_make_spawner(" in inspect.getsource(run_cell), "run_cell bypasses the spawner"


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

    # Read from `bench.warmup`: the warmup moved out of the script so `_execution_digest` reaches
    # it, and the gate that consumes this key still lives in the script. The two modules are
    # exactly where the key can drift, so the assertion has to span them.
    emitted = inspect.getsource(run_warmup)
    assert '"warmups"' in emitted, "warmup payload key drifted from the gate that reads it"


def test_results_carry_the_engine_shape_that_produced_them() -> None:
    """Without the resolved catalog row, a capacity number cannot be tied to the checkpoint,
    context limit or sequence cap behind it, and catalog drift silently reinterprets old results."""
    source = BENCH_APP.read_text()
    assert source.count("bench_catalog_summary") >= 2, "provenance helper is still test-only"

    bucket_payload = source[source.index("async def _run_bucket(") :]
    bucket_payload = bucket_payload[: bucket_payload.index("\ndef _write_artifact")]
    assert '"engine_catalog"' in bucket_payload, "per-bucket payload omits the engine shape"

    # The envelope lives in `_Lane.write`, so EVERY artifact carries the engine shape and no
    # individual write site can forget it. Assert it there, and that both lanes write through it.
    lane_write = source[source.index("    def write(") :]
    lane_write = lane_write[: lane_write.index("\n    def settle(")]
    assert '"engine_catalog"' in lane_write, "the artifact envelope omits the engine shape"
    for lane_fn in ("_run_canary_lane", "_run_sweep_lane"):
        body = source[source.index(f"def {lane_fn}(") :]
        body = body[: body.index("\n\n\n")]
        assert "lane.write(" in body, f"{lane_fn} bypasses the envelope that carries the shape"


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

    # And the driver actually supplies the expectation rather than discarding it. The spawn lives
    # in `_make_spawner`, which is the only place `run_cell` creates requests from.
    from flash.serving.bench.driver import _make_spawner

    assert "expected_prompt_tokens=exact" in inspect.getsource(_make_spawner), (
        "the fitted length is discarded at spawn"
    )


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
    # Derived from real bench code rather than transcribed, so a change to how the script bounds a
    # warmup fit still moves this expectation.
    warmup_fit = prompt_fit_seconds_bound(BUCKETS_BY_NAME["short_interactive"], min_requests=1)
    namespace.update(
        bench_engine_overrides_for=bench_engine_overrides_for,
        concurrency_grid=concurrency_grid,
        prompt_fit_seconds_bound=prompt_fit_seconds_bound,
        REQUEST_TIMEOUT_SECONDS=REQUEST_TIMEOUT_SECONDS,
        WARMUP_FIT_SECONDS_BOUND=warmup_fit,
        _FUNDED_WARMUP_FIT_SECONDS=warmup_fit + fitting_watchdog_grace_seconds(),
        # The count moved into `bench.warmup` so `_execution_digest` reaches it; the script imports
        # it, so the lifted estimator resolves it from here rather than from a lifted assignment.
        CANARY_WARMUP_REQUESTS=CANARY_WARMUP_REQUESTS,
        fitting_watchdog_grace_seconds=fitting_watchdog_grace_seconds,
        drain_reap_seconds=drain_reap_seconds,
        run_request_bound_seconds=run_request_bound_seconds,
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<bench>", "exec"), namespace)
    estimate_for = namespace["_sweep_gpu_seconds_estimate"]

    bucket = BUCKETS_BY_NAME["short_interactive"]
    estimate = estimate_for("Qwen/Qwen3.5-9B", [bucket])

    points = len(list(concurrency_grid(8)))
    windows = float(bucket.max_seconds) * points
    # Billed on the container but outside every `max_seconds`, so it is reserved separately. Each
    # cell's pool arms its own watchdog, which ends the container at `bound + grace`, so the grace
    # is reserved once per cell alongside the fitting it guards.
    fitting = (prompt_fit_seconds_bound(bucket) + fitting_watchdog_grace_seconds()) * points
    # A drain waits its own timeout and then up to `drain_reap_seconds()` for a task that ignored
    # cancellation. Both are bounded and both bill, so pricing only the first left the reap interval
    # funded by nothing at every single concurrency point.
    drains = (REQUEST_TIMEOUT_SECONDS + drain_reap_seconds()) * points
    # Each warmup pays its fit as well as its request: the fit runs outside `run_request` and is
    # billed on the same container, so pricing a warmup at the request timeout alone left an
    # enforced-but-unfunded phase. Priced at the funded fit, which includes the watchdog grace.
    # The canary's warmups await through the driver's ENFORCED bound, which permits the request
    # timeout plus a reap for a stream that ignored cancellation. Both are billed.
    canary = (
        run_request_bound_seconds() + warmup_fit + fitting_watchdog_grace_seconds()
    ) * CANARY_WARMUP_REQUESTS
    startup = constants["STARTUP_TIMEOUT_SECONDS"]
    # `max_containers=1` caps SIMULTANEOUS replicas; it does not pin successive `.remote()` calls to
    # one container. Every bucket call can therefore land on a cold replacement and pay another boot
    # plus its warmups, which bills whether or not the reservation admits it.
    replacements = (startup + canary) * 1
    # No separate canary replacement boot: `_run_canary` is ONE remote call (`certify.remote()`
    # probes and warms the SAME container), so there is no gap between two calls for a replacement
    # to land in. Reserving one anyway would refuse runs that cannot cost that much.
    # The canary's probe plus one per bucket: `_run_bucket` gates the container it measured on, and
    # each probe is bounded by `PROBE_TIMEOUT_SECONDS` rather than the class method timeout.
    probes = constants["PROBE_TIMEOUT_SECONDS"] * 2
    # `min_containers=0` does not release the GPU at the last return -- each separately bootable
    # call can leave a container alive and billing for its whole scaledown window.
    scaledown = float(constants["SCALEDOWN_WINDOW_SECONDS"]) * 2
    # Modal enforces `TIMEOUT_SECONDS`, which is the worst-case bucket PLUS this headroom, so each
    # separately bootable call may bill the slack on top of every phase above before being killed.
    headroom = float(constants["TIMEOUT_HEADROOM_SECONDS"]) * 2

    assert estimate >= windows + drains + canary, (
        "the estimate omits the canary or the per-cell drain tails"
    )
    # The boot is reserved at the ceiling Modal lets a stuck boot reach, not a typical observed one.
    assert estimate == pytest.approx(
        startup + canary + probes + windows + fitting + drains + replacements + scaledown + headroom
    )


def test_a_bucket_landing_on_a_cold_container_warms_itself_before_measuring() -> None:
    """`max_containers=1` caps simultaneous replicas; it does not pin successive calls to one.

    So a bucket can land on a container the canary never gated, and would otherwise pay compile and
    lazy-workspace costs inside its measured window.
    """
    from flash.serving.bench import driver as driver_module

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
    assert "run_warmup" in warm_called, "the cold path does not actually warm up"
    # Read through `getattr` with a default, since a freshly booted container has no such
    # attribute at all, so the name appears as a string constant rather than an attribute access.
    read_names = {
        child.attr for child in ast.walk(nodes["_ensure_warm"]) if isinstance(child, ast.Attribute)
    } | {
        child.value
        for child in ast.walk(nodes["_ensure_warm"])
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }
    assert "_bench_warmed" in read_names, "warmth is not tracked per container"

    # The flag is SET by the warmup that knows whether it worked, not by the caller that merely
    # asked for one; see the failed-warmup guard. Assert the reader still short-circuits on it and
    # that exactly one place writes it, so a second writer cannot reintroduce an unvalidated warm.
    def _writers(module: ast.Module) -> set[str]:
        return {
            node.name
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and any(
                isinstance(target, ast.Attribute) and target.attr == "_bench_warmed"
                for child in ast.walk(node)
                if isinstance(child, ast.Assign)
                for target in child.targets
            )
        }

    # Checked across ALL THREE modules because the warmup moved out of the script and into
    # `bench.warmup`: a script-only scan now finds no writer at all and would pass vacuously,
    # including on a version where some caller in the driver sets the flag itself.
    from flash.serving.bench import warmup as warmup_module

    script_writers = _writers(tree)
    driver_writers = _writers(ast.parse(inspect.getsource(driver_module)))
    warmup_writers = _writers(ast.parse(inspect.getsource(warmup_module)))
    assert script_writers == set(), (
        f"the script writes the warmed flag from {sorted(script_writers)}; only the warmup that "
        "knows whether generation worked may set it"
    )
    assert driver_writers == set(), (
        f"the driver writes the warmed flag from {sorted(driver_writers)}; only the warmup that "
        "knows whether generation worked may set it"
    )
    assert warmup_writers == {"run_warmup"}, (
        f"the warmed flag is written by {sorted(warmup_writers)}; exactly one writer, the warmup "
        "itself, may set it, or a caller can mark a container warm that no warmup validated"
    )


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
    # `WARMUP_FIT_SECONDS_BOUND` is a module-level assignment whose RHS calls into the bench
    # package, so lifting that node requires its dependencies in scope. Provided here rather than at
    # each call site: the value is derived from real bench code either way, so the test still fails
    # if the script changes how the bound is computed.
    namespace: dict[str, Any] = {
        "Any": Any,
        "uuid": uuid,
        "BUCKETS_BY_NAME": BUCKETS_BY_NAME,
        "prompt_fit_seconds_bound": prompt_fit_seconds_bound,
        # The REAL accessors, not stand-in numbers. Both report intervals the driver enforces, so a
        # test that hardcoded them would keep passing after the driver retuned one -- which is the
        # exact drift between what is enforced and what is reserved these tests exist to catch.
        "fitting_watchdog_grace_seconds": fitting_watchdog_grace_seconds,
        "drain_reap_seconds": drain_reap_seconds,
        # Moved into the driver so `_execution_digest` reaches the warmup contract. The script
        # imports them, so they are no longer liftable nodes -- supplied here as the REAL driver
        # objects, which keeps every lifted caller running against the code that actually ships.
        "CANARY_WARMUP_REQUESTS": CANARY_WARMUP_REQUESTS,
        "warmup_fit_seconds_bound": warmup_fit_seconds_bound,
        "run_warmup": run_warmup,
        # The warmup awaits its request through the driver's enforced bound, which permits the
        # request timeout plus a reap for a stream that ignored cancellation. Reserving the
        # nominal timeout instead would leave that reap funded by nothing.
        "run_request_bound_seconds": run_request_bound_seconds,
        **injected,
    }
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
        "bucket_call_priced_seconds",
        "_worst_case_bucket_seconds",
        "TIMEOUT_HEADROOM_SECONDS",
        "TIMEOUT_SECONDS",
        "WARMUP_FIT_SECONDS_BOUND",
        "_FUNDED_WARMUP_FIT_SECONDS",
        "SCALEDOWN_WINDOW_SECONDS",
        "PROBE_TIMEOUT_SECONDS",
        BENCH_MODELS=BENCH_MODELS,
        bench_engine_overrides_for=bench_engine_overrides_for,
        concurrency_grid=concurrency_grid,
        BUCKETS=BUCKETS,
        REQUEST_TIMEOUT_SECONDS=REQUEST_TIMEOUT_SECONDS,
        prompt_fit_seconds_bound=prompt_fit_seconds_bound,
    )
    timeout = namespace["TIMEOUT_SECONDS"]

    # Recompute the worst case independently of the script's own helper.
    points = max(
        len(list(concurrency_grid(int(bench_engine_overrides_for(model).get("max_num_seqs", 8)))))
        for model in BENCH_MODELS
    )
    # Maximized PER BUCKET. Taking the widest window and the widest fitting separately would
    # describe a bucket that does not exist; the timeout has to fit the worst real one.
    #
    # Prompt fitting is inside this bound even though it is outside `max_seconds`: it runs in the
    # container before each cell's window opens, so the method clock runs through it. The budget
    # estimator already reserved it; omitting it here left the same phase unbounded in the place
    # that actually terminates the call.
    #
    # The two bounded TAILS belong here too. A drain waits its own timeout and then up to
    # `drain_reap_seconds()` for a task that ignored cancellation, and a pool's watchdog ends the
    # container at `bound + grace`, not at the bound. Both are time the code still permits, so a
    # method timeout priced at the nominal halves can fire while a legitimate phase is running --
    # killing `run_bucket` before it writes the artifact the run already paid for.
    widest = max(
        bucket.max_seconds
        + REQUEST_TIMEOUT_SECONDS
        + drain_reap_seconds()
        + prompt_fit_seconds_bound(bucket)
        + fitting_watchdog_grace_seconds()
        for bucket in BUCKETS
    )
    # A bucket that lands on a cold replacement warms it SEQUENTIALLY before the first cell. Those
    # warmups run inside the method, so a timeout sized to the grid alone kills the call mid-warmup
    # -- and because the timeout fires before `run_bucket` returns, the artifact is never written.
    # Each of those warmups also FITS a prompt first, under its own watchdog, so the method clock
    # runs through the funded fit as well as the request.
    warmups = (
        run_request_bound_seconds() + namespace["_FUNDED_WARMUP_FIT_SECONDS"]
    ) * CANARY_WARMUP_REQUESTS
    worst = points * widest + warmups + namespace["PROBE_TIMEOUT_SECONDS"]
    assert namespace["_worst_case_bucket_seconds"]() == worst
    # Guard the direction that costs money: the grid's own fitting must be INSIDE the bound.
    assert (
        worst
        > points * (max(bucket.max_seconds for bucket in BUCKETS) + REQUEST_TIMEOUT_SECONDS)
        + warmups
    ), "the timeout ignores prompt fitting, which runs on the method clock"
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

    # Called directly: the warmup lives in `bench.warmup` now, so there is nothing to lift.
    engine = mock.Mock(tokenizer=object(), base_model="Qwen/Qwen3.5-9B")
    with (
        mock.patch("flash.serving.bench.driver.run_request", _fake_run_request),
        mock.patch("flash.serving.bench.warmup.fit_prompt_to_tokens", _fake_fit),
    ):
        asyncio.run(run_warmup(engine, 5))
        first = list(issued)
        issued.clear()
        asyncio.run(run_warmup(engine, 5))

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
        "WARMUP_FIT_SECONDS_BOUND",
        "_FUNDED_WARMUP_FIT_SECONDS",
        "SCALEDOWN_WINDOW_SECONDS",
        run_warmup=_fake_run_warmup,
    )
    engine = mock.Mock()
    del engine._bench_warmed  # a freshly booted container has never been warmed

    with pytest.raises(RuntimeError, match="replacement-container warmup failed 1/2"):
        asyncio.run(namespace["_ensure_warm"](engine))
    assert calls, "the cold path never ran a warmup at all"


def test_a_failed_warmup_leaves_the_container_unwarmed_so_a_retry_reruns_it() -> None:
    """A rejected warmup must not leave the container flagged warm.

    ``_ensure_warm`` short-circuits on ``_bench_warmed``, so whoever sets that flag decides whether
    a later bucket warms at all. Setting it on entry to ``run_warmup`` records INTENT: a warmup
    whose records come back ``ok=False`` -- or that raises part way, since the fit watchdog and the
    request bound both raise -- still leaves the flag true. ``_ensure_warm`` raises and the attempt
    is refused, but the flag outlives the refusal, so a retry inside the scaledown window lands on
    the same container, short-circuits at the flag check, and measures a generation path no warmup
    ever validated. Both failure shapes are checked because they reach the flag differently.
    """

    class _Record:
        def __init__(self, ok: bool) -> None:
            self._ok = ok

        def to_json(self) -> dict[str, Any]:
            return {"ok": self._ok, "error": None if self._ok else "cache_contaminated"}

    def _fake_fit(tokenizer: Any, uid: str, target: int) -> tuple[list[dict[str, str]], int]:
        return ([{"role": "user", "content": uid}], target)

    # `run_warmup` is NOT faked here: the flag write under test is inside it, and the helper's
    # defaults supply the real `bench.warmup` function the lifted `_ensure_warm` resolves.
    namespace = _bench_namespace("_ensure_warm", "_require_healthy_warmup")

    # 1. Records returned ok=False: the warmup completes, so only an outcome check can catch it.
    async def _unhealthy(
        engine: Any, base_model: str, messages: Any, max_tokens: int, uid: str, **kw: Any
    ) -> Any:
        return _Record(ok=False)

    engine = mock.Mock(tokenizer=object(), base_model="Qwen/Qwen3.5-9B")
    del engine._bench_warmed  # a freshly booted container has never been warmed
    with (
        mock.patch("flash.serving.bench.driver.run_request", _unhealthy),
        mock.patch("flash.serving.bench.warmup.fit_prompt_to_tokens", _fake_fit),
        pytest.raises(RuntimeError, match="replacement-container warmup failed"),
    ):
        asyncio.run(namespace["_ensure_warm"](engine))
    assert getattr(engine, "_bench_warmed", False) is False, (
        "a warmup whose records failed left the container flagged warm, so a retry inside the "
        "scaledown window short-circuits and measures an engine no warmup ever validated"
    )

    # 2. The warmup raises part way, the shape the fit watchdog and the request bound produce.
    async def _raises(
        engine: Any, base_model: str, messages: Any, max_tokens: int, uid: str, **kw: Any
    ) -> Any:
        raise TimeoutError("stream close hung past the enforced bound")

    engine = mock.Mock(tokenizer=object(), base_model="Qwen/Qwen3.5-9B")
    del engine._bench_warmed
    with (
        mock.patch("flash.serving.bench.driver.run_request", _raises),
        mock.patch("flash.serving.bench.warmup.fit_prompt_to_tokens", _fake_fit),
        pytest.raises(TimeoutError),
    ):
        asyncio.run(namespace["_ensure_warm"](engine))
    assert getattr(engine, "_bench_warmed", False) is False, (
        "a warmup that raised left the container flagged warm"
    )

    # A healthy warmup must still mark the container, or every bucket repays the one-time costs.
    async def _healthy(
        engine: Any, base_model: str, messages: Any, max_tokens: int, uid: str, **kw: Any
    ) -> Any:
        return _Record(ok=True)

    engine = mock.Mock(tokenizer=object(), base_model="Qwen/Qwen3.5-9B")
    del engine._bench_warmed
    with (
        mock.patch("flash.serving.bench.driver.run_request", _healthy),
        mock.patch("flash.serving.bench.warmup.fit_prompt_to_tokens", _fake_fit),
    ):
        assert asyncio.run(namespace["_ensure_warm"](engine)) is not None
    assert engine._bench_warmed is True, "a healthy warmup no longer marks the container warm"
    assert asyncio.run(namespace["_ensure_warm"](engine)) is None, "the warm container re-warmed"


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
        "WARMUP_FIT_SECONDS_BOUND",
        "_FUNDED_WARMUP_FIT_SECONDS",
        "TIMEOUT_HEADROOM_SECONDS",
        "SCALEDOWN_WINDOW_SECONDS",
        "STARTUP_TIMEOUT_SECONDS",
        "PROBE_TIMEOUT_SECONDS",
        bench_engine_overrides_for=bench_engine_overrides_for,
        concurrency_grid=concurrency_grid,
        prompt_fit_seconds_bound=prompt_fit_seconds_bound,
        REQUEST_TIMEOUT_SECONDS=REQUEST_TIMEOUT_SECONDS,
    )
    startup = namespace["STARTUP_TIMEOUT_SECONDS"]
    warmups = CANARY_WARMUP_REQUESTS
    # A warmup bills its prompt fit as well as its request: the fit runs outside `run_request`, on
    # the same rented container. Priced at the FUNDED figure, not the nominal bound: the watchdog
    # ends the container at `bound + grace`, so the grace is billable time the lane permits itself.
    # The warmup awaits through the driver's ENFORCED bound, not the nominal request timeout:
    # a stream that ignores cancellation costs a further reap, and it is billed.
    per_warmup = run_request_bound_seconds() + namespace["_FUNDED_WARMUP_FIT_SECONDS"]
    assert namespace["_FUNDED_WARMUP_FIT_SECONDS"] == pytest.approx(
        namespace["WARMUP_FIT_SECONDS_BOUND"] + fitting_watchdog_grace_seconds()
    ), "the funded warmup fit dropped the watchdog grace it is supposed to cover"
    # The container keeps billing through its scaledown window after the last call returns.
    scaledown = float(namespace["SCALEDOWN_WINDOW_SECONDS"])
    # `TIMEOUT_SECONDS` exceeds the phases priced here by this much, and Modal enforces the larger
    # number, so every separately bootable call may bill the slack before being terminated.
    headroom = float(namespace["TIMEOUT_HEADROOM_SECONDS"])

    canary = namespace["_canary_gpu_seconds_estimate"]()
    # ONE boot: `certify.remote()` probes and warms the same container in a single call, so a
    # replacement cannot land between a probe and a warmup and no second boot is billable.
    # The probe is still reserved: it runs inside that call, AFTER `@enter`, so it is billed on top
    # of the boot rather than overlapping it, and it is bounded so a stall cannot reach the class
    # method timeout.
    assert (
        canary
        == startup * 1
        + namespace["PROBE_TIMEOUT_SECONDS"]
        + (per_warmup * warmups)
        + scaledown
        + headroom * 1
    )
    assert canary >= startup, "the canary reserves less than a stuck boot alone would bill"

    bucket = BUCKETS_BY_NAME["short_interactive"]
    sweep = namespace["_sweep_gpu_seconds_estimate"]("Qwen/Qwen3.5-9B", [bucket])
    points = len(list(concurrency_grid(8)))
    expected = (
        startup
        # No second canary boot: `certify.remote()` probes and warms one container, so there is no
        # gap between two calls for a replacement to land in.
        + per_warmup * warmups
        + bucket.max_seconds * points
        # Prompt fitting runs on the rented container before each cell's window opens, so it is
        # billed even though it is deliberately outside `max_seconds`. Each cell's pool arms its own
        # watchdog, which ends the container at `bound + grace`, so the grace is reserved per cell.
        + (prompt_fit_seconds_bound(bucket) + fitting_watchdog_grace_seconds()) * points
        # Each drain waits its timeout and then up to `drain_reap_seconds()` for a task that ignored
        # cancellation; both are bounded and both bill.
        + (REQUEST_TIMEOUT_SECONDS + drain_reap_seconds()) * points
        # one replacement boot + warmup per bucket call, since `max_containers=1` does not pin
        # successive `.remote()` calls to the container the previous bucket booted
        + (startup + per_warmup * warmups) * 1
        # the canary's probe plus one per bucket: `_run_bucket` gates the container it measures on
        + namespace["PROBE_TIMEOUT_SECONDS"] * 2
        # one scaledown tail per separately bootable call
        + scaledown * 2
        # one method-timeout grant per separately bootable call, for the same reason
        + headroom * 2
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
        "WARMUP_FIT_SECONDS_BOUND",
        "_FUNDED_WARMUP_FIT_SECONDS",
        "TIMEOUT_HEADROOM_SECONDS",
        "SCALEDOWN_WINDOW_SECONDS",
        "STARTUP_TIMEOUT_SECONDS",
        "PROBE_TIMEOUT_SECONDS",
        bench_engine_overrides_for=bench_engine_overrides_for,
        concurrency_grid=concurrency_grid,
        prompt_fit_seconds_bound=prompt_fit_seconds_bound,
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


def test_workload_checksum_covers_prompt_construction_not_just_token_counts() -> None:
    """Two campaigns can agree on every token count and still measure different work.

    The digest previously named only bucket dimensions and decoding, so the filler vocabulary, the
    fit tolerance, and the concurrency grid could all change without moving it -- and a published
    result would claim a preregistered workload it did not run. Each is mutated independently here,
    because a digest that happens to include one of them still lets the other two drift.
    """
    baseline = workload_checksum()

    with mock.patch("flash.serving.bench.workload.PROMPT_TOKEN_TOLERANCE", 999):
        assert workload_checksum() != baseline, "the fit tolerance is not in the checksum"

    with mock.patch(
        "flash.serving.bench.workload._FILLER_WORDS", ("alpha", "beta", "gamma", "delta")
    ):
        assert workload_checksum() != baseline, "the filler vocabulary is not in the checksum"

    def _narrower_grid(max_num_seqs: int) -> tuple[int, ...]:
        return (1, max_num_seqs)

    with mock.patch("flash.serving.bench.workload.concurrency_grid", _narrower_grid):
        assert workload_checksum() != baseline, "the concurrency grid is not in the checksum"

    # And the digest is still stable when nothing moves, so the test above is detecting the mutation
    # rather than nondeterminism in the digest itself.
    assert workload_checksum() == baseline


def test_documented_rate_denominator_matches_the_one_reduce_cell_uses() -> None:
    """The doc must describe the denominator the code divides by.

    An earlier round moved rates onto the pre-drain window but left the prose saying "full wall
    time", which is the kind of drift a reader cannot detect from either artifact alone. Rather than
    grep for a phrase, this computes a cell whose drain would change the answer and asserts the code
    excludes it -- then asserts the doc does not describe the excluded behaviour.
    """
    records = [
        _record("in-window", latency=10.0, tokens=100),
        # Finishes AFTER the 20s window closed: a drain completion.
        _record("in-drain", latency=45.0, tokens=100),
    ]
    result = reduce_cell(
        records,
        base_model="Qwen/Qwen3.5-9B",
        bucket="short_interactive",
        concurrency=2,
        block=0,
        wall_seconds=20.0,
        drain_seconds=25.0,
        window_seconds=20.0,
    )
    assert result.succeeded == 2, "both requests did succeed"
    assert result.succeeded_in_window == 1, "only one finished inside the window"
    # 1 in-window success over the 20s window, NOT 2 over 45s.
    assert result.successful_rps == pytest.approx(1 / 20.0)
    assert result.output_tokens_per_second == pytest.approx(100 / 20.0)

    doc = (REPO_ROOT / "docs" / "serving-capacity-envelope.md").read_text(encoding="utf-8")
    assert "full wall time" not in doc, (
        "the doc still describes rates over full wall time, but reduce_cell divides by the "
        "pre-drain window"
    )
    assert "steady-state window" in doc


def _fake_gdn_modules(leaf: Any) -> dict[str, Any]:
    """Every ancestor package, because ``from a.b.c import d`` imports ``a.b.c`` first.

    Registering only the leaf leaves the import raising ModuleNotFoundError, the resolver unreached,
    and a test that would pass against a probe which never calls it.
    """
    parent = "vllm.model_executor.layers.mamba.gdn"
    modules: dict[str, Any] = {}
    parts = parent.split(".")
    for depth in range(1, len(parts) + 1):
        modules[".".join(parts[:depth])] = types.ModuleType(".".join(parts[:depth]))
    modules[parent].qwen_gdn_linear_attn = leaf
    modules[f"{parent}.qwen_gdn_linear_attn"] = leaf
    return modules


def test_gdn_probe_calls_the_vllm_config_resolver_with_served_geometry() -> None:
    """The pinned vLLM resolver takes one config object, not raw head dimensions."""
    seen: dict[str, Any] = {}

    def _resolver(vllm_config: Any) -> tuple[str, str]:
        seen["additional_config"] = vllm_config.additional_config
        seen["head_dim"] = vllm_config.model_config.hf_text_config.linear_key_head_dim
        return "auto", "cutedsl"

    from flash.serving.bench.probe import _gdn_backend_in_process

    module = mock.Mock()
    module._resolve_gdn_prefill_backend = _resolver
    with (
        mock.patch.dict("sys.modules", _fake_gdn_modules(module)),
        mock.patch(
            "flash.serving.bench.probe._gdn_config_values",
            return_value={"linear_key_head_dim": 96},
        ),
        mock.patch(
            "flash.serving.bench.probe.probe_cutlass_integrity",
            return_value={"checked": False},
        ),
    ):
        result = _gdn_backend_in_process("Qwen/Qwen3.5-9B")

    assert seen == {"additional_config": None, "head_dim": 96}
    assert result["resolved"] == "cutedsl"
    assert result["resolved_raw"] == ["auto", "cutedsl"]
    assert result["source"] == "resolver"
    assert not result.get("resolver_signature_mismatch")


def test_gdn_probe_still_binds_raw_head_dimension_resolvers() -> None:
    """Signature-driven binding remains valid for builds that declare raw geometry parameters."""
    seen: list[int] = []

    def _resolver(linear_key_head_dim: int) -> str:
        seen.append(linear_key_head_dim)
        return "flashinfer"

    from flash.serving.bench.probe import _gdn_backend_in_process

    module = mock.Mock()
    module._resolve_gdn_prefill_backend = _resolver
    with (
        mock.patch.dict("sys.modules", _fake_gdn_modules(module)),
        mock.patch(
            "flash.serving.bench.probe._gdn_config_values",
            return_value={"linear_key_head_dim": 80},
        ),
        mock.patch(
            "flash.serving.bench.probe.probe_cutlass_integrity",
            return_value={"checked": False},
        ),
    ):
        result = _gdn_backend_in_process("Qwen/Qwen3.5-9B")

    assert seen == [80]
    assert result["resolved"] == "flashinfer"


def test_gdn_probe_records_an_unknown_required_parameter_as_a_signature_mismatch() -> None:
    """An unsupported required parameter remains distinct from an unknown backend decision."""

    def _needs_unknown(some_new_parameter: int) -> str:  # pragma: no cover - never called
        return "flashinfer"

    from flash.serving.bench.probe import _gdn_backend_in_process

    module = mock.Mock()
    module._resolve_gdn_prefill_backend = _needs_unknown
    with (
        mock.patch.dict("sys.modules", _fake_gdn_modules(module)),
        mock.patch("flash.serving.bench.probe._gdn_config_values", return_value={}),
        mock.patch(
            "flash.serving.bench.probe.probe_cutlass_integrity",
            return_value={"checked": False},
        ),
    ):
        result = _gdn_backend_in_process("Qwen/Qwen3.5-9B")

    assert result["resolved"] is None
    assert result["resolver_signature_mismatch"] is True
    assert "signature unsupported" in result["reason"]
    assert "some_new_parameter" in result["reason"]


def test_resolver_binding_refuses_a_config_with_no_gdn_head_dimension() -> None:
    """A missing head dim must fail, not default.

    On SM10.x the resolver grants FlashInfer only when ``linear_key_head_dim == 128``. Passing
    ``None`` therefore returns "triton" -- indistinguishable from the real Blackwell fallback this
    probe exists to catch, but manufactured by the probe itself. The failed config read has to
    surface as a signature mismatch so the gate refuses the run.
    """
    from flash.serving.bench.probe import _gdn_backend_in_process

    def _needs_vllm_config(vllm_config: Any) -> tuple[str, str]:
        raise AssertionError("resolver must not be called without a real head dimension")

    module = mock.Mock()
    module._resolve_gdn_prefill_backend = _needs_vllm_config
    with (
        mock.patch.dict("sys.modules", _fake_gdn_modules(module)),
        mock.patch("flash.serving.bench.probe._gdn_config_values", return_value={}),
        mock.patch(
            "flash.serving.bench.probe.probe_cutlass_integrity",
            return_value={"checked": False},
        ),
    ):
        result = _gdn_backend_in_process("Qwen/Qwen3.5-9B")

    assert result["resolved"] is None
    assert result["resolver_signature_mismatch"] is True
    assert "linear_key_head_dim" in result["reason"]


def test_probe_all_reads_the_gdn_backend_from_the_isolated_resolver() -> None:
    """The engine object is not a GDN evidence source.

    vLLM V1 executes the model in a separate EngineCore process (see the 2026-07-05 post-mortem in
    ``lora_engine.py``), so no layer carrying ``gdn_prefill_backend`` is reachable from this
    process. The subprocess resolver is the only path that can answer, and it must be consulted
    even when a live engine is passed.
    """
    from flash.serving.bench import probe as probe_module

    engine = types.SimpleNamespace(engine=types.SimpleNamespace())
    resolver_result = {"resolved": "cutedsl", "source": "resolver"}
    with (
        mock.patch.object(probe_module, "probe_resolved_revisions", return_value={}),
        mock.patch.object(probe_module, "probe_runtime_packages", return_value={}),
        mock.patch.object(probe_module, "probe_gpu", return_value={}),
        mock.patch.object(
            probe_module, "probe_gdn_backend", return_value=resolver_result.copy()
        ) as resolver_probe,
        mock.patch.object(probe_module, "probe_engine_kv_cache", return_value={}),
    ):
        result = probe_module.probe_all("Qwen/Qwen3.6-35B-A3B", engine)

    resolver_probe.assert_called_once_with("Qwen/Qwen3.6-35B-A3B")
    assert result["gdn_prefill"]["resolved"] == "cutedsl"
    assert result["gdn_prefill"]["source"] == "resolver"


def test_the_paid_entrypoint_settles_its_reservation() -> None:
    """A published budget that only ever shows the worst case is not a cost report.

    ``reserve`` is deliberately generous so the ceiling is safe; if nothing settles it, every
    artifact reports that generous number as the spend. The entrypoint must call ``settle`` on both
    lanes, so this asserts against the SOURCE that the calls exist on each path.
    """
    source = BENCH_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "main"
    )
    # Settlement is centralized in `_Lane.settle`; the three lanes call it. Counting `lane.settle`
    # across the module covers the canary lane and sweep lane helpers as well as the failure path.
    del main
    settles = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "lane.settle"
    ]
    # Three: the canary lane, the sweep lane, and the failure handler that settles a lane which
    # died mid-flight. A failed lane has already spent its boot and probe, so leaving it unsettled
    # would under-report the campaign's committed spend exactly where the accounting matters most.
    assert len(settles) == 3, (
        "expected the canary lane, the sweep lane, and the failure path each to settle their "
        f"reservation; found {len(settles)}"
    )
    # And the reservation is bound, not discarded -- `settle` needs the entry `reserve` returned.
    assert re.search(r"entry\s*=\s*ledger\.reserve\(", source), (
        "the ledger entry is not captured, so nothing can be settled against it"
    )


def test_settling_replaces_the_reservation_in_the_committed_total() -> None:
    """Settlement has to MOVE the committed total, not merely annotate the entry."""
    ledger = BudgetLedger(ceiling_usd=50.0)
    entry = ledger.reserve("sweep:Qwen/Qwen3.5-9B", 9000.0, "L40S")
    reserved_total = ledger.committed_usd
    assert reserved_total > 0

    # The lane actually ran a tenth of its worst case.
    ledger.settle(entry, 900.0, note="measured sweep wall")
    assert entry.gpu_seconds == 900.0
    assert entry.settled_usd == pytest.approx(usd_for_gpu_seconds(900.0, "L40S"))
    assert ledger.committed_usd < reserved_total, (
        "committed_usd still reports the reservation after settlement"
    )
    assert ledger.committed_usd == pytest.approx(entry.settled_usd)


def test_prompt_pool_period_does_not_depend_on_concurrency():
    """R7: two points on ONE curve must send the same corpus sequence.

    Sizing the pool as ``min_requests + concurrency`` moved the wrap point per point, so request 301
    reused corpus 0 at c=1 but corpus 301 at c=16. Completion lengths then differed between points
    for a reason that is not offered load, and the derived knee moved with it.
    """
    bucket = BUCKETS_BY_NAME["short_interactive"]
    low, low_calls = _fake_pool(concurrency=1)
    high, high_calls = _fake_pool(concurrency=16)

    assert len(low) == len(high) == bucket.min_requests + _POOL_PERIOD_SLACK
    # The corpus seeds are the load-bearing part: they are what the wrap point used to shift.
    assert [corpus for _, corpus in low_calls] == [corpus for _, corpus in high_calls]


def test_prompt_fit_raises_rather_than_publishing_an_out_of_band_prompt():
    """R7: a prompt the search cannot fit is a workload defect, not a number to publish.

    The driver validates the engine's reported length against the FITTED count, so a near-miss
    transmits faithfully and validates cleanly while the published bucket label is wrong.
    """
    from flash.serving.bench.workload import PromptFitError, fit_prompt_to_tokens

    class _StuckTokenizer:
        """Always reports the same length, so the search can never converge."""

        def apply_chat_template(self, messages, **kwargs):
            return "x"

        def __call__(self, text, **kwargs):
            # One token regardless of input, so no guess can approach a 4096-token target.
            return {"input_ids": [0]}

    with pytest.raises(PromptFitError):
        fit_prompt_to_tokens(_StuckTokenizer(), "uid", 4096)


def test_prompt_fit_seconds_bound_scales_with_input_size_and_depth():
    """R7: fitting is billed GPU wall time and must be reserved.

    It is excluded from ``max_seconds`` on purpose -- tokenization must not compete with the
    measured window -- so the reservation is the only place it can be funded.
    """
    short = prompt_fit_seconds_bound(BUCKETS_BY_NAME["short_interactive"])
    near = prompt_fit_seconds_bound(BUCKETS_BY_NAME["near_32k"])
    assert short > 0.0
    # 31744 input tokens per prompt dwarfs 512, even though near_32k pools far fewer prompts.
    assert near > short


# The script's own warmup-fit bound, computed from real bench code so a change to how the script
# bounds a single-prompt fit still moves every expectation derived from it.
_WARMUP_FIT_BOUND = prompt_fit_seconds_bound(BUCKETS_BY_NAME["short_interactive"], min_requests=1)


def test_sweep_estimate_includes_prompt_fitting():
    """R7: the reservation must exceed the same sweep priced without the fitting term."""
    namespace = _bench_namespace(
        "_sweep_gpu_seconds_estimate",
        "TIMEOUT_HEADROOM_SECONDS",
        "SCALEDOWN_WINDOW_SECONDS",
        "STARTUP_TIMEOUT_SECONDS",
        "PROBE_TIMEOUT_SECONDS",
        REQUEST_TIMEOUT_SECONDS=REQUEST_TIMEOUT_SECONDS,
        bench_engine_overrides_for=bench_engine_overrides_for,
        concurrency_grid=concurrency_grid,
        prompt_fit_seconds_bound=prompt_fit_seconds_bound,
        # Held FIXED on both sides rather than lifted: recomputing it through the stub below would
        # zero the canary's fitting term too, and the difference would stop isolating cell fitting.
        WARMUP_FIT_SECONDS_BOUND=_WARMUP_FIT_BOUND,
        _FUNDED_WARMUP_FIT_SECONDS=_WARMUP_FIT_BOUND + fitting_watchdog_grace_seconds(),
    )
    model = "Qwen/Qwen3.5-9B"
    selected = [BUCKETS_BY_NAME["near_32k"]]
    total = namespace["_sweep_gpu_seconds_estimate"](model, selected)

    points = len(
        list(concurrency_grid(int(bench_engine_overrides_for(model).get("max_num_seqs", 8))))
    )
    fitting = prompt_fit_seconds_bound(selected[0]) * points
    assert fitting > 0.0

    # Prove the term is genuinely INSIDE the total, not merely smaller than it: re-price the same
    # sweep with fitting stubbed to zero and require the difference to be exactly the fitting.
    zeroed = _bench_namespace(
        "_sweep_gpu_seconds_estimate",
        "TIMEOUT_HEADROOM_SECONDS",
        "SCALEDOWN_WINDOW_SECONDS",
        "STARTUP_TIMEOUT_SECONDS",
        "PROBE_TIMEOUT_SECONDS",
        REQUEST_TIMEOUT_SECONDS=REQUEST_TIMEOUT_SECONDS,
        bench_engine_overrides_for=bench_engine_overrides_for,
        concurrency_grid=concurrency_grid,
        prompt_fit_seconds_bound=lambda bucket, **kw: 0.0,
        WARMUP_FIT_SECONDS_BOUND=_WARMUP_FIT_BOUND,
        _FUNDED_WARMUP_FIT_SECONDS=_WARMUP_FIT_BOUND + fitting_watchdog_grace_seconds(),
    )["_sweep_gpu_seconds_estimate"](model, selected)
    # Only the per-cell FIT is stubbed away; the watchdog grace reserved beside it is a constant the
    # stub cannot zero, so it cancels in the difference and the isolation stays exact.
    assert total - zeroed == pytest.approx(fitting)


def test_gdn_probe_reads_the_served_checkpoint_not_the_logical_name():
    """R7: geometry must come from the weights the engine loaded.

    The 9B and 27B engines serve separate ``-FP8`` repositories and the 27B pins a
    ``model_revision``. Probing the mutable logical name could report a backend for a checkpoint
    nobody served once that repository advances.
    """
    from flash.serving.bench.catalog import bench_engine_overrides_for
    from flash.serving.bench.probe import _served_checkpoint

    for base_model in BENCH_MODELS:
        overrides = bench_engine_overrides_for(base_model)
        repo, revision = _served_checkpoint(base_model)
        assert repo == overrides["serve_model_id"]
        assert revision == overrides.get("model_revision")


def test_early_stop_gates_on_steady_state_successes():
    """R7: a cell whose successes all land in the DRAIN publishes zero throughput.

    Gating the climb on ``succeeded`` kept it positive there, so the sweep bought another full
    window-and-drain tail per remaining point after the cell was already classified degraded.
    """
    from flash.serving.bench.driver import grid_should_halt

    source = inspect.getsource(grid_should_halt)
    assert "result.succeeded_in_window == 0" in source
    assert "result.succeeded == 0" not in source
    # The script must CONSULT it, not carry its own copy. A predicate nothing calls would leave the
    # inline policy in place while this test passed against the unused helper.
    script = BENCH_APP.read_text(encoding="utf-8")
    assert "if grid_should_halt(result):" in script
    assert "if result.succeeded_in_window == 0:" not in script

    # And it must actually decide. `succeeded` stays positive in the halting case on purpose: that
    # is the exact cell -- every success landed in the drain -- that gating on `succeeded` kept
    # climbing past.
    def _at(in_window: int) -> CellResult:
        return CellResult(
            base_model="Qwen/Qwen3.5-9B",
            bucket="short_interactive",
            concurrency=8,
            block=0,
            wall_seconds=60.0,
            attempted=10,
            succeeded=4,
            succeeded_in_window=in_window,
            failed=6,
        )

    assert grid_should_halt(_at(0)) is True
    assert grid_should_halt(_at(1)) is False


def test_canary_gate_refuses_an_unresolved_gdn_backend() -> None:
    """An unknown kernel path must stop the lane, not be waved through by a healthy warmup.

    `probe_gdn_backend` records `resolved=None` when the resolver is missing, its signature moved,
    or the served config would not load. None of those stop the engine booting, serving, and
    billing, so without this gate a sweep could be paid for and published with no evidence of which
    GDN prefill kernel produced the numbers -- the one label that makes a Blackwell result
    interpretable.
    """
    # Assert the CALL SITE, not only the predicate. A helper that nothing invokes is exactly the
    # reported defect -- `_run_canary` gated on card identity and Cutlass integrity and never read
    # `resolved` -- so a test that only exercises the predicate passes against the broken gate.
    source = BENCH_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
    }
    called = {
        child.func.id
        for child in ast.walk(nodes["_run_canary"])
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }
    assert "_require_resolved_gdn_backend" in called, (
        "the canary no longer checks which GDN backend resolved"
    )

    namespace = _bench_namespace("_require_resolved_gdn_backend")
    require = namespace["_require_resolved_gdn_backend"]

    # A resolved backend passes.
    require({"gdn_prefill": {"resolved": "flashinfer"}})

    # Unresolved for any reason must raise, and the reason must survive into the message.
    for probe, expected in (
        (
            {"gdn_prefill": {"resolved": None, "reason": "vllm build has no resolver"}},
            "no resolver",
        ),
        (
            {
                "gdn_prefill": {
                    "resolved": None,
                    "reason": "resolver requires 'linear_key_head_dim'",
                    "resolver_signature_mismatch": True,
                }
            },
            "signature mismatch",
        ),
        ({"gdn_prefill": {}}, "no reason"),
        ({}, "no reason"),
    ):
        with pytest.raises(RuntimeError, match="unresolved"):
            require(probe)
        with pytest.raises(RuntimeError, match=expected):
            require(probe)


def test_sweep_reserves_one_boot_per_separately_bootable_call() -> None:
    """The sweep makes `len(buckets) + 1` separately bootable calls, so it reserves that many boots.

    `max_containers=1` caps simultaneous replicas without pinning successive calls to one container,
    so every bucket call can pay its own cold boot. The canary contributes ONE: `certify.remote()`
    probes and warms the same container, so unlike the old `probe`/`warmup` pair there is no gap for
    a replacement to land in. A reservation must be wrong toward refusing a run, but reserving a
    boot that cannot happen refuses runs that would have fit.
    """
    namespace = _bench_namespace(
        "_sweep_gpu_seconds_estimate",
        "WARMUP_FIT_SECONDS_BOUND",
        "_FUNDED_WARMUP_FIT_SECONDS",
        "TIMEOUT_HEADROOM_SECONDS",
        "SCALEDOWN_WINDOW_SECONDS",
        "STARTUP_TIMEOUT_SECONDS",
        "PROBE_TIMEOUT_SECONDS",
        bench_engine_overrides_for=bench_engine_overrides_for,
        concurrency_grid=concurrency_grid,
        prompt_fit_seconds_bound=prompt_fit_seconds_bound,
        REQUEST_TIMEOUT_SECONDS=REQUEST_TIMEOUT_SECONDS,
    )
    startup = namespace["STARTUP_TIMEOUT_SECONDS"]
    estimate_for = namespace["_sweep_gpu_seconds_estimate"]

    for count in (1, 2, 3):
        selected = list(BUCKETS)[:count]
        estimate = estimate_for("Qwen/Qwen3.5-9B", selected)
        points = len(list(concurrency_grid(8)))
        windows = sum(float(b.max_seconds) * points for b in selected)
        # Per cell, the fit plus the grace its watchdog terminates at.
        fitting = sum(
            (prompt_fit_seconds_bound(b) + fitting_watchdog_grace_seconds()) * points
            for b in selected
        )
        # Per cell, the drain timeout plus the reap a task that ignored cancellation costs.
        drains = (REQUEST_TIMEOUT_SECONDS + drain_reap_seconds()) * points * count
        # A warmup bills its prompt fit as well as its request; the fit runs outside `run_request`,
        # and is priced at the funded figure because its watchdog ends at `bound + grace`.
        warmups = (
            run_request_bound_seconds() + namespace["_FUNDED_WARMUP_FIT_SECONDS"]
        ) * CANARY_WARMUP_REQUESTS
        # Boots reserved: initial + canary replacement + one per bucket call.
        boots = startup * (count + 1)
        # Each separately bootable call can leave a container billing through its scaledown window.
        scaledown = float(namespace["SCALEDOWN_WINDOW_SECONDS"]) * (count + 1)
        # One bounded probe per bucket call plus the canary's own, each capped by its own bound
        # rather than by the class method timeout.
        probes = namespace["PROBE_TIMEOUT_SECONDS"] * (count + 1)
        # Each call is enforced at `TIMEOUT_SECONDS`, which exceeds the phases above by this much;
        # the slack is billable before Modal terminates the call, so it is reserved per call.
        headroom = float(namespace["TIMEOUT_HEADROOM_SECONDS"]) * (count + 1)
        expected = (
            boots
            + probes
            + warmups * (count + 1)
            + windows
            + fitting
            + drains
            + scaledown
            + headroom
        )
        assert estimate == pytest.approx(expected), (
            f"{count}-bucket sweep must reserve {count + 1} boots"
        )


def test_each_bucket_gates_the_container_it_actually_measures_on() -> None:
    """The GDN gate must run on the bucket's OWN container, before its first cell.

    `max_containers=1` caps simultaneous replicas without pinning successive `.remote()` calls to
    one container, so the canary's probe can describe a container the bucket never touches. The
    provenance block was read only AFTER the whole grid and serialized ungated, so an unresolved
    kernel path still produced a publishable artifact -- paid for in full, and attributable to no
    kernel.

    Asserted on the CALL SITE, not the predicate. A gate nothing invokes is precisely the reported
    defect, and a test that only exercises `_require_resolved_gdn_backend` in isolation passes
    against it.
    """
    source = BENCH_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
    }
    run_bucket = nodes["_run_bucket"]
    called = {
        child.func.id
        for child in ast.walk(run_bucket)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }
    # Reached through the shared gate. The bucket used to invoke `_require_resolved_gdn_backend`
    # itself; the same three gates now live in `_gate_container_provenance` so that no caller can
    # hold a probe without applying it. The property under test is unchanged -- this bucket refuses
    # an unresolved kernel path before it measures -- so resolve the call through the helper rather
    # than pinning a call name the policy no longer keeps here.
    assert "_gate_container_provenance" in called, (
        "a bucket can measure and publish on a container whose kernel path was never established"
    )
    helper = nodes["_gate_container_provenance"]
    assert any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "_require_resolved_gdn_backend"
        for child in ast.walk(helper)
    ), "the shared gate no longer establishes the kernel path"

    # The gate must precede the first cell, or it refuses nothing that was not already paid for.
    body = run_bucket.body
    gate_line = min(
        child.lineno
        for child in ast.walk(run_bucket)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "_gate_container_provenance"
    )
    loop_line = min(child.lineno for child in body if isinstance(child, ast.For | ast.AsyncFor))
    assert gate_line < loop_line, "the kernel gate runs after cells the run has already paid for"

    # And the artifact must carry the probe the gate ACCEPTED. Re-probing at serialization time
    # would report a container state nothing refused.
    assert "_probe_in_container_within_bound" in called, (
        "the in-container probe is unbounded; it can outrun the bucket's own reservation"
    )
    body_src = ast.get_source_segment(source, run_bucket) or ""
    assert body_src.count("_probe_in_container_within_bound(engine)") == 1, (
        "the bucket probes twice; the published provenance is then not the gated one"
    )
    # Card identity on THIS container too, not only the kernel path: a replacement container on a
    # different accelerator would otherwise be published under the catalog's expected GPU label.
    # The check now travels inside the shared gate, whose call site is already asserted above to
    # precede the first cell, so the card is verified before anything is paid for.
    assert any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "gpu_matches"
        for child in ast.walk(helper)
    ), "a bucket can publish a measurement taken on an unverified card"


def test_the_probe_call_is_bounded_below_the_class_method_timeout() -> None:
    """A stalled probe must not bill the whole method timeout.

    `probe.remote()` inherits the class `timeout`, which is sized for an entire concurrency grid.
    The probe only reads NVML, asks vLLM's resolver for its GDN choice, and loads the served
    config, so a probe approaching that ceiling is stalled -- but on the class bound it would bill
    hours against a lane whose estimate reserved it nothing.
    """
    source = BENCH_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
    }
    # The canary must certify ONE container: a probe and a warmup issued as two remote calls could
    # land on different containers, and the artifact would then pair container A's accepted
    # provenance with container B's generation health.
    canary_src = ast.get_source_segment(source, nodes["_run_canary"]) or ""
    assert "certify.remote(" in canary_src, (
        "the canary probes and warms separately; it cannot certify one container"
    )
    assert "warmup.remote(" not in canary_src, (
        "a second remote call reopens the split between probed and measured containers"
    )

    namespace = _bench_namespace(
        "PROBE_TIMEOUT_SECONDS",
        "STARTUP_TIMEOUT_SECONDS",
        "TIMEOUT_SECONDS",
        "TIMEOUT_HEADROOM_SECONDS",
        "WARMUP_FIT_SECONDS_BOUND",
        "_FUNDED_WARMUP_FIT_SECONDS",
        "SCALEDOWN_WINDOW_SECONDS",
        "bucket_call_priced_seconds",
        "_worst_case_bucket_seconds",
        BENCH_MODELS=BENCH_MODELS,
        bench_engine_overrides_for=bench_engine_overrides_for,
        concurrency_grid=concurrency_grid,
        BUCKETS=BUCKETS,
        REQUEST_TIMEOUT_SECONDS=REQUEST_TIMEOUT_SECONDS,
        prompt_fit_seconds_bound=prompt_fit_seconds_bound,
    )
    bound = namespace["PROBE_TIMEOUT_SECONDS"]
    assert 0 < bound < namespace["TIMEOUT_SECONDS"], (
        "a probe bound at or above the method timeout bounds nothing"
    )

    # The bound must cover probe WORK only. Bounding a spawned call timed the cold `@enter` model
    # load too, and the 35B/H200 engine takes roughly 17 minutes to initialize -- a 300s bound would
    # have cancelled its first probe every time, despite the class allowing a 2700s startup. Inside
    # the already-loaded method, boot is governed by `startup_timeout` instead.
    assert bound < namespace["STARTUP_TIMEOUT_SECONDS"], (
        "a probe bound above the startup allowance would never bind probe work"
    )
    certify_src = ast.get_source_segment(source, nodes["_build_bench_engine"]) or ""
    assert "_probe_in_container_within_bound(self)" in certify_src, (
        "the probe is bounded outside the container, so its bound also times the cold boot"
    )


def test_the_module_runbook_ceilings_clear_their_own_submission_stop() -> None:
    """The script's own docstring is a runbook too, and it drifted from the doc's.

    `test_documented_ceilings_exceed_what_each_lane_reserves` checks the markdown. The module
    docstring carries the same two commands, and after the fitting and second-boot reservations
    landed it still said `--ceiling-usd 16` -- copy-pasteable, and refused before allocation by the
    lane's own 80% submission stop.
    """
    namespace = _bench_namespace(
        "_canary_gpu_seconds_estimate",
        "_sweep_gpu_seconds_estimate",
        "WARMUP_FIT_SECONDS_BOUND",
        "_FUNDED_WARMUP_FIT_SECONDS",
        "TIMEOUT_HEADROOM_SECONDS",
        "SCALEDOWN_WINDOW_SECONDS",
        "STARTUP_TIMEOUT_SECONDS",
        "PROBE_TIMEOUT_SECONDS",
        bench_engine_overrides_for=bench_engine_overrides_for,
        concurrency_grid=concurrency_grid,
        prompt_fit_seconds_bound=prompt_fit_seconds_bound,
        REQUEST_TIMEOUT_SECONDS=REQUEST_TIMEOUT_SECONDS,
    )
    docstring = ast.get_docstring(ast.parse(BENCH_APP.read_text(encoding="utf-8"))) or ""
    documented = {
        match.group(1): float(match.group(2))
        for match in re.finditer(
            r"--mode (canary|sweep)[\s\S]*?--ceiling-usd (\d+(?:\.\d+)?)", docstring
        )
    }
    assert set(documented) == {"canary", "sweep"}, "the module runbook lost one of its commands"

    model = "Qwen/Qwen3.5-9B"
    gpu = bench_gpu_for(model)
    reserved = {
        "canary": namespace["_canary_gpu_seconds_estimate"](),
        "sweep": namespace["_sweep_gpu_seconds_estimate"](
            model, [BUCKETS_BY_NAME["short_interactive"]]
        ),
    }
    for mode, seconds in reserved.items():
        required = usd_for_gpu_seconds(seconds, gpu) / SUBMISSION_STOP_FRACTION
        assert documented[mode] >= required, (
            f"the module docstring's --ceiling-usd {documented[mode]} for the {mode} lane is "
            f"below the ${required:.2f} its reservation needs to clear the submission stop"
        )


def test_the_in_container_probe_is_bounded_by_the_probe_allowance() -> None:
    """`_run_bucket` runs INSIDE the container, so `probe_all` there is a local call.

    `_probe_within_bound` cannot cover it -- that helper spawns a remote call. But it is the same
    stall with the same funding: both estimators reserve exactly `PROBE_TIMEOUT_SECONDS` per bucket,
    and an unbounded in-process probe would run to the class-wide method timeout, overrun the
    bucket's reservation, and lose the whole paid artifact when Modal terminates `run_bucket`.

    Bounded via a worker thread, because `probe_all` is synchronous: awaited directly it would pin
    the event loop where no timeout could fire.
    """
    source = BENCH_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
    }
    helper = nodes["_probe_in_container_within_bound"]
    src = ast.get_source_segment(source, helper) or ""
    assert "wait_for" in src, "an unbounded in-process probe cannot be interrupted"
    assert "timeout=PROBE_TIMEOUT_SECONDS" in src, (
        "the probe is bounded by something other than the allowance both estimators reserve"
    )
    assert "run_in_executor" in src, (
        "a synchronous probe awaited on the event loop cannot be timed out"
    )


def test_an_explicit_zero_submission_stop_permits_no_spend() -> None:
    """`submission_stop_usd=0` is the emergency stop, not an unset value.

    The guard read `self.submission_stop_usd or self.ceiling_usd`, so an explicit zero was falsy and
    restored the FULL ceiling -- admitting reservations in the one configuration whose entire
    purpose is to admit none.
    """
    from flash.serving.bench.budget import BudgetExceeded, BudgetLedger

    ledger = BudgetLedger(ceiling_usd=100.0, submission_stop_usd=0.0)
    assert ledger.submission_stop_usd == 0.0, "an explicit stop must survive construction"
    assert not ledger.can_submit(1.0, "L40S"), "a zero stop permits no new submission"
    with pytest.raises(BudgetExceeded):
        ledger.reserve("any", 1.0, "L40S")

    with pytest.raises(ValueError, match="must not be negative"):
        BudgetLedger(ceiling_usd=100.0, submission_stop_usd=-1.0)


def test_the_workload_checksum_covers_the_construction_implementation() -> None:
    """Editing prompt construction must move the checksum.

    An enumeration of inputs (filler vocabulary, tolerance, grid) covers only what someone
    remembered to list. `_prompt_header`, `build_prompt_text`, `corpus_seed`, `reseed_prompt` and
    the fitting logic all change what the model reads or how a prompt is assembled, and every one
    of them sat outside the digest -- so two materially different workload contracts could publish
    under one checksum.
    """
    from flash.serving.bench import workload

    assert workload._CONSTRUCTION_SOURCES, "no construction source is digested"
    for name in (
        "_prompt_header",
        "build_prompt_text",
        "corpus_seed",
        "reseed_prompt",
        "fit_prompt_to_tokens",
    ):
        assert name in workload._CONSTRUCTION_SOURCES, (
            f"{name} can change without moving the digest"
        )

    before = workload.workload_checksum()
    original = workload._prompt_header

    def _edited(uid: str) -> str:
        return "DIFFERENT" + original(uid)

    workload._prompt_header = _edited
    try:
        assert workload.workload_checksum() != before, (
            "prompt construction changed but the checksum did not"
        )
    finally:
        workload._prompt_header = original
    assert workload.workload_checksum() == before, "the digest is not stable across restoration"


def test_the_catalog_row_carries_the_whole_resolved_override_mapping() -> None:
    """A curve is only interpretable against the exact engine shape that produced it.

    The row promoted a hand-picked subset, omitting capacity-defining settings such as
    `gpu_memory_utilization`, `enforce_eager`, `max_num_batched_tokens` and `pin_loras`. Once
    production drifts, an artifact holding only the subset still looks plausible while no longer
    identifying the shape it measured.
    """
    from flash.serving.bench.catalog import bench_catalog_summary, bench_engine_overrides_for

    for row in bench_catalog_summary():
        assert row["engine_overrides"] == bench_engine_overrides_for(row["base_model"]), (
            f"{row['base_model']} publishes a subset of the overrides it actually ran"
        )


def test_a_reseeded_prompt_is_validated_against_the_bucket_target() -> None:
    """A wrapped request must not drift two tolerances out of its published bucket.

    `reseed_prompt` rewrites the header without re-tokenizing, so the record still carries the
    POOLED prompt's fitted count. Checking only against that stale value allows another full
    tolerance on top of a fit that may already sit a tolerance from the target, so a request up to
    `2 * PROMPT_TOKEN_TOLERANCE` out of bucket could be counted in it. The bucket label is the
    published claim, so the engine's own count has to satisfy it.
    """
    import inspect

    from flash.serving.bench import driver
    from flash.serving.bench.metrics import RequestRecord
    from flash.serving.bench.workload import PROMPT_TOKEN_TOLERANCE

    assert "bucket_target_tokens" in inspect.signature(driver.run_request).parameters, (
        "the bucket target never reaches the record, so it cannot be validated"
    )
    assert "bucket_target_tokens" in RequestRecord.__dataclass_fields__

    validate = driver._validate
    target = 512
    # Fitted a full tolerance BELOW target; engine reports a full tolerance below the fit. Within
    # tolerance of `expected`, but 2x out from the bucket the number would be published under.
    fitted = target - PROMPT_TOKEN_TOLERANCE
    reported = fitted - PROMPT_TOKEN_TOLERANCE
    record = RequestRecord(
        uid="u",
        base_model="Qwen/Qwen3.5-9B",
        bucket="short_interactive",
        concurrency=1,
        block=0,
        started_at=0.0,
        expected_prompt_tokens=fitted,
        bucket_target_tokens=target,
    )
    outcome = driver._StreamOutcome()
    outcome.saw_final = True
    outcome.finish_reason = "stop"
    outcome.prompt_tokens = reported
    outcome.completion_tokens = 32
    outcome.cached_tokens = 0
    outcome.cached_tokens_reported = True
    outcome.first_token_at = 0.1
    validate(outcome, record)
    assert not record.ok, "a request two tolerances out of bucket was counted in it"
    assert record.error == driver.ERROR_PROMPT_LENGTH


def test_a_timed_out_in_container_probe_terminates_the_container() -> None:
    """Abandoning the future does not stop the thread; only ending the process does.

    `run_in_executor` hands `probe_all` to a worker thread, and a thread blocked in
    `AutoConfig.from_pretrained` or an NVML C call cannot be interrupted from Python. Raising on
    timeout fails the call but leaves the container alive through its scaledown window with the
    stuck thread still on the GPU -- still billing past the reservation the bound exists to enforce,
    and still available to a retry that would inherit it. A spawned call had
    `terminate_containers=True` for exactly this; from inside, the equivalent is to exit.
    """
    source = BENCH_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
    }
    src = ast.get_source_segment(source, nodes["_probe_in_container_within_bound"]) or ""
    assert "os._exit" in src, (
        "the timed-out probe keeps billing: nothing terminates the container it is stuck in"
    )
    # A raise would be caught by the same event loop the stuck thread is already outliving, so the
    # handler must not merely raise.
    handler = next(
        node
        for node in ast.walk(nodes["_probe_in_container_within_bound"])
        if isinstance(node, ast.ExceptHandler)
    )
    assert not any(isinstance(child, ast.Raise) for child in ast.walk(handler)), (
        "raising leaves the container alive with an uninterruptible thread on the GPU"
    )


def test_the_probe_payload_survives_a_local_process_without_the_serving_image() -> None:
    """A probe value typed by torch loses the run it already paid for.

    `probe_all` returns across a Modal boundary into a LOCAL process that has no torch, no vllm and
    no transformers. `torch.__version__` is a `str` SUBCLASS defined in `torch.torch_version`, so it
    pickles by reference to that module and the local unpickle dies with `ModuleNotFoundError:
    torch` -- after the container has booted the engine, run the probe and billed for the GPU. The
    canary on 2026-08-31 lost exactly that way: a healthy L40S boot with no verdict to show for it.

    The failure is reproduced with a subclass whose defining module genuinely is not importable
    here, which is the same shape as the real one rather than a stand-in for it.
    """
    import pickle

    from flash.serving.bench import probe as probe_module

    namespace: dict[str, Any] = {}
    exec(compile("class Version(str):\n    pass\n", "absent_pkg/_version.py", "exec"), namespace)
    version_type = namespace["Version"]
    version_type.__module__ = "absent_pkg._version"
    foreign = version_type("2.11.0+cu130")

    # The premise: this value really is unpicklable outside the image that defines its type.
    with pytest.raises(pickle.PicklingError, match=r"absent_pkg\.\_version"):
        pickle.loads(pickle.dumps(foreign))

    plain = probe_module._plain_types({"gpu": {"torch_version": foreign, "count": 1, "ok": True}})
    assert pickle.loads(pickle.dumps(plain)) == {
        "gpu": {"torch_version": "2.11.0+cu130", "count": 1, "ok": True}
    }, "a foreign-typed probe value still crosses the boundary and loses the run"
    assert type(plain["gpu"]["torch_version"]) is str, (
        "the value is still its subclass: `isinstance` passes it while pickle still needs the module"
    )
    # Ints and bools must not be flattened to strings on the way through -- the summary compares
    # block counts and reads flags, so coercing everything would trade one broken artifact for
    # another.
    assert type(plain["gpu"]["count"]) is int
    assert type(plain["gpu"]["ok"]) is bool

    # And the real probe must actually route through it, not merely define it.
    source = Path(probe_module.__file__).read_text(encoding="utf-8")
    probe_all_body = source[source.index("def probe_all(") : source.index("def _plain_types(")]
    assert "return _plain_types(payload)" in probe_all_body, (
        "probe_all returns its payload unguarded: a foreign-typed field added later loses a run"
    )


def test_a_cancelled_in_container_probe_terminates_the_container_too() -> None:
    """Cancelling the wait is not catching it: `CancelledError` is not a `TimeoutError`.

    `CancelledError` derives from `BaseException`, so a handler that names only `TimeoutError` lets
    it straight past. The asyncio future is then cancelled while the executor thread and its
    resolver subprocess keep running; `_within_call_bound` sees its task already cancelled and
    re-raises without terminating anything; and the container survives its scaledown window with an
    uninterruptible thread on the GPU, billing on and inheritable by a later invocation. A thread
    Python cannot interrupt is unreapable whichever way the wait ended, so both endings must end the
    process -- the same argument the timeout case already makes.
    """
    source = BENCH_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
    }
    handler = next(
        node
        for node in ast.walk(nodes["_probe_in_container_within_bound"])
        if isinstance(node, ast.ExceptHandler)
    )
    caught = {
        ast.unparse(node)
        for node in (handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type])
    }
    assert "asyncio.CancelledError" in caught, (
        "a cancelled probe escapes the handler: the container keeps billing with an "
        f"uninterruptible thread on the GPU (handler catches {sorted(caught)})"
    )
    # Catching it is only half the fix -- the handler must still end the process rather than fall
    # through to a return, which would report a cancelled probe as a successful one.
    src = ast.get_source_segment(source, nodes["_probe_in_container_within_bound"]) or ""
    assert "os._exit" in src, "the cancelled probe returns past its bound without terminating"


def test_the_canary_certifies_one_container_in_one_remote_call() -> None:
    """Provenance and generation health must describe the SAME container.

    `max_containers=1` caps simultaneous replicas without pinning successive calls to one container,
    so a probe call and a warmup call could land on different containers. The canary would then
    accept container A's GDN backend and card while reporting container B's warmup records -- and B
    is free to hold an unresolved backend or a different accelerator.
    """
    source = BENCH_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
    }
    canary_src = ast.get_source_segment(source, nodes["_run_canary"]) or ""
    assert canary_src.count(".remote(") == 1, (
        "the canary makes more than one remote call, so a replacement can split what it certifies"
    )
    assert "certify.remote(" in canary_src

    factory = ast.get_source_segment(source, nodes["_build_bench_engine"]) or ""
    assert "async def certify(" in factory, "no single method probes and warms one container"
    # Probe THEN warm, so a container that fails its gate never pays for warmup requests.
    assert factory.index("_probe_in_container_within_bound(self)") < factory.index(
        "run_warmup(self, requests)"
    ), "the canary warms before it probes, paying for a container it may reject"


def test_the_probe_bound_does_not_time_the_cold_boot() -> None:
    """The 300s allowance must cover probe work, not model loading.

    Bounding a spawned call timed the whole invocation including the class `@enter` load. The
    35B/H200 engine needs roughly 17 minutes to initialize, so its first probe would have been
    cancelled every time despite `startup_timeout=STARTUP_TIMEOUT_SECONDS` permitting 2700s.
    """
    namespace = _bench_namespace("PROBE_TIMEOUT_SECONDS", "STARTUP_TIMEOUT_SECONDS")
    assert namespace["PROBE_TIMEOUT_SECONDS"] < namespace["STARTUP_TIMEOUT_SECONDS"], (
        "a probe bound at or above the startup allowance cannot bind probe work alone"
    )
    source = BENCH_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
    }
    # Bounded from INSIDE the loaded method. A spawned-and-bounded probe would reintroduce the
    # boot-inclusive timer wholesale, so the helper that did it must stay gone.
    assert "_probe_within_bound" not in nodes, (
        "the spawned probe helper is back; its bound times the cold boot too"
    )
    factory = ast.get_source_segment(source, nodes["_build_bench_engine"]) or ""
    assert "_probe_in_container_within_bound(self)" in factory


def test_the_construction_digest_covers_the_helpers_the_prompts_call() -> None:
    """`inspect.getsource` reads a function's own text, never its callees.

    `_deterministic_words` decides every filler body and `request_uid` decides every header and
    cache key, so an edit to either changes the workload materially -- while changing no source that
    was being digested. Two different workloads would publish under one checksum.
    """
    from flash.serving.bench import workload

    for name in ("_deterministic_words", "request_uid"):
        assert name in workload._CONSTRUCTION_SOURCES, (
            f"{name} can change without moving the digest"
        )

    before = workload.workload_checksum()
    original = workload._deterministic_words

    def _edited(seed_material: str, count: int) -> list[str]:
        return ["different"] * count

    workload._deterministic_words = _edited
    try:
        assert workload.workload_checksum() != before, (
            "the filler-body generator changed but the checksum did not"
        )
    finally:
        workload._deterministic_words = original
    assert workload.workload_checksum() == before


def test_a_retry_cannot_destroy_the_prior_invocations_artifacts(tmp_path) -> None:
    """`write_text` truncates, and every artifact name repeats on a retry.

    The filenames key on model, bucket and block -- all of which a rerun repeats -- so a second run
    silently overwrote the first run's per-bucket evidence and its summary. That inverts the point
    of the invocation nonce, which exists to make retries distinguishable; their evidence is the
    part worth distinguishing.
    """
    namespace = _bench_namespace("_write_artifact", os=os, json=json, Path=Path)
    write = namespace["_write_artifact"]

    os.environ["BENCH_OUT_DIR"] = str(tmp_path)
    try:
        first = write({"run": 1}, "summary-model-b0.json", invocation="aaaaaaaaaaaa")
        second = write({"run": 2}, "summary-model-b0.json", invocation="bbbbbbbbbbbb")
        assert first != second, "two invocations collided on one path"
        assert json.loads(first.read_text())["run"] == 1, "the first run's evidence was destroyed"
        assert json.loads(second.read_text())["run"] == 2

        # Even within one invocation, a repeated name must refuse rather than truncate.
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            write({"run": 3}, "summary-model-b0.json", invocation="aaaaaaaaaaaa")
    finally:
        os.environ.pop("BENCH_OUT_DIR", None)


def test_a_failed_write_leaves_no_artifact_under_the_real_name(tmp_path) -> None:
    """A paid artifact must appear whole or not at all.

    The remote bucket has already run and already been billed by the time the write executes, so a
    truncated JSON file under the artifact's own name is indistinguishable later from a bucket that
    measured a short curve -- and the artifact is the only surviving evidence of what was paid for.
    The failure is injected DURING the write, not in serialization: `json.dumps` runs to completion
    before any file is opened, so a dumps failure never created a file even on the old path and
    proves nothing. A sink that dies part way through does -- an interrupted write leaves bytes on
    disk that no longer parse. Under the atomic write those bytes are in a temp file that is removed
    and never renamed, so the real name stays absent.
    """
    namespace = _bench_namespace(
        "_write_artifact", os=os, json=json, Path=Path, contextlib=contextlib
    )
    write = namespace["_write_artifact"]

    real_open = builtins.open

    class _DiesMidWrite:
        """Writes the first chunk, then fails the way a full disk or a signal would."""

        def __init__(self, handle):
            self._handle = handle
            self._wrote = False

        def write(self, data):
            # The payload is one `json.dumps` string, so this is called ONCE: land half the bytes
            # on disk, then fail. That is the shape an interrupted write actually has.
            self._handle.write(data[: max(1, len(data) // 2)])
            self._handle.flush()
            self._wrote = True
            raise OSError(28, "No space left on device")

        def __getattr__(self, name):
            return getattr(self._handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

    def _failing_open(*args, **kwargs):
        return _DiesMidWrite(real_open(*args, **kwargs))

    payload = {"rows": ["x" * 100] * 5000}
    os.environ["BENCH_OUT_DIR"] = str(tmp_path)
    try:
        namespace["open"] = _failing_open
        with pytest.raises(OSError, match="No space left on device"):
            write(payload, "summary-model-b0.json", invocation="aaaaaaaaaaaa")
        namespace["open"] = real_open
        out_dir = tmp_path / "aaaaaaaaaaaa"
        leftovers = sorted(p.name for p in out_dir.iterdir()) if out_dir.exists() else []
        assert leftovers == [], f"a failed write left files behind: {leftovers}"

        # And the happy path still lands the whole payload under the real name.
        path = write({"run": 1}, "summary-model-b0.json", invocation="aaaaaaaaaaaa")
        assert json.loads(path.read_text())["run"] == 1
        assert sorted(p.name for p in path.parent.iterdir()) == ["summary-model-b0.json"]
    finally:
        os.environ.pop("BENCH_OUT_DIR", None)


def test_the_workload_checksum_covers_the_driver_and_metric_contract() -> None:
    """A driver retune or a metric threshold change must move the checksum.

    The construction digest answers "which prompts"; it says nothing about which prompts get
    ISSUED, which attempts become errors, or where a curve is declared saturated. Two campaigns
    could differ in the pool period, the request timeout or the saturation thresholds and still
    publish the same contract identifier, which is exactly the claim a checksum exists to prevent.

    Constants are checked separately from function sources on purpose: a constant's NAME is all
    that appears in the text of the function reading it, so a pure retune leaves every source
    digest byte-identical.
    """
    from flash.serving.bench import driver, metrics
    from flash.serving.bench import workload as workload_module

    before = workload_module.workload_checksum()

    # A metric threshold, carried as a keyword default rather than a module constant.
    original_reduce = metrics.reduce_cell

    def _retuned(*args: Any, **kwargs: Any) -> Any:
        return original_reduce(*args, **kwargs)

    metrics.reduce_cell = _retuned
    try:
        assert workload_module.workload_checksum() != before, (
            "the reduction implementation is not in the checksum"
        )
    finally:
        metrics.reduce_cell = original_reduce
    assert workload_module.workload_checksum() == before, "the digest is not stable across restore"

    # A pure constant retune: no function source changes at all.
    original_slack = driver._POOL_PERIOD_SLACK
    driver._POOL_PERIOD_SLACK = original_slack + 1
    try:
        assert workload_module.workload_checksum() != before, (
            "_POOL_PERIOD_SLACK is not in the checksum, so a pool retune is invisible"
        )
    finally:
        driver._POOL_PERIOD_SLACK = original_slack

    original_timeout = driver.REQUEST_TIMEOUT_SECONDS
    driver.REQUEST_TIMEOUT_SECONDS = original_timeout + 1.0
    try:
        assert workload_module.workload_checksum() != before, (
            "REQUEST_TIMEOUT_SECONDS is not in the checksum, so what counts as a timeout can move"
        )
    finally:
        driver.REQUEST_TIMEOUT_SECONDS = original_timeout

    assert workload_module.workload_checksum() == before, "the digest is not stable across restore"


def test_the_construction_and_execution_digests_are_reported_separately() -> None:
    """Prompt contract and execution contract must be distinguishable in the material.

    Collapsing them into one value would make a driver retune indistinguishable from a prompt
    change when a stale artifact has to be explained, which is the whole reason the digest is
    recorded rather than a bare version number.
    """
    from flash.serving.bench import workload as workload_module

    source = inspect.getsource(workload_module.workload_checksum)
    assert "construction=" in source
    assert "execution=" in source
    assert workload_module._construction_digest() != workload_module._execution_digest()


def test_every_bucket_artifact_carries_its_own_workload_checksum() -> None:
    """Per-bucket files are written eagerly, so each must be self-describing.

    They exist precisely because a later bucket can fail; at that point they are the only surviving
    evidence of a boot that was already paid for. A payload with measurements but no contract
    identifier cannot say which prompts and reduction rules produced it.
    """
    source = BENCH_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    sweep = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_run_sweep_lane"
    )
    bucket_writes = [
        node
        for node in ast.walk(sweep)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "lane.write"
        and any("sweep-" in ast.unparse(arg) for arg in node.args)
    ]
    assert bucket_writes, "no per-bucket artifact write found"
    for call in bucket_writes:
        assert ast.unparse(call.args[0]) == "payload", (
            "the bucket artifact no longer carries the remote payload"
        )
    # The identities are FROZEN before the first remote call and spread by the envelope, so the
    # checksum reaches every artifact through the mapping rather than a call at the write site.
    # Recomputing it at the write site is the defect, not the contract.
    write = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "write"
    )
    assert "**self.local_identity" in ast.unparse(write), (
        "a per-bucket artifact is persisted without its workload checksum"
    )


def test_a_failed_paid_lane_settles_and_records_its_spend() -> None:
    """A lane that dies mid-sweep has already spent; the accounting must survive it.

    The reservation and elapsed wall live only in the local process, so without this the boot,
    probe and cells a failed lane paid for leave no evidence and the campaign's committed spend is
    silently under-reported. The handler must account and RE-RAISE, never rescue.
    """
    source = BENCH_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    handlers = [node for node in ast.walk(main) if isinstance(node, ast.Try)]
    assert handlers, "the paid section is not wrapped in failure handling"

    guarding = []
    for try_node in handlers:
        body = ast.unparse(try_node)
        if "_run_canary" in body and "_run_sweep_lane" in body:
            guarding.append(try_node)
    assert guarding, "the wrapped region does not cover both the canary and the bucket calls"

    for try_node in guarding:
        for handler in try_node.handlers:
            text = ast.unparse(handler)
            assert "lane.settle" in text, "a failed lane does not settle its elapsed spend"
            assert "lane.write" in text, "a failed lane writes no accounting artifact"
            assert any(
                isinstance(node, ast.Raise) and node.exc is None for node in ast.walk(handler)
            ), "the failure handler swallows the exception instead of re-raising it"
            # BaseException, not Exception: a Modal timeout or an interrupt spends exactly the same
            # GPU-seconds as an error does.
            assert "BaseException" in ast.unparse(handler.type), (
                "the handler misses interrupts and timeouts, which bill like any other failure"
            )


def test_prompt_fitting_enforces_the_bound_it_reserves() -> None:
    """The fitting loop must obey `prompt_fit_seconds_bound`, not merely be estimated by it.

    Unenforced, a slow tokenizer keeps billing against the class's shared `TIMEOUT_SECONDS` --
    which for a short-only sweep is sized for `near_32k`, i.e. far more GPU time than the lane
    reserved. It must END THE PROCESS rather than raise, matching the probe bound: a raise unwinds
    into a container that is still billing.
    """
    from flash.serving.bench import driver as driver_module

    source = inspect.getsource(driver_module._build_prompt_pool)
    assert "prompt_fit_seconds_bound" in source, (
        "the pool does not derive a deadline from its bound"
    )
    assert "os._exit" in source, "an exceeded fitting bound does not end the container"

    tree = ast.parse(textwrap.dedent(source))
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    guards = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.If) and "deadline" in ast.unparse(node.test)
    ]
    assert guards, "no deadline check inside the fitting loop"
    for guard in guards:
        assert not [node for node in ast.walk(guard) if isinstance(node, ast.Raise)], (
            "the deadline raises instead of exiting, so the container keeps billing while unwinding"
        )


def test_provenance_reads_the_snapshot_each_role_was_actually_loaded_from() -> None:
    """``refs/<rev>`` is per-REPOSITORY; provenance is per-ROLE, and the two diverge mid-boot.

    The 35B serves its weights and its tokenizer from ONE unpinned repository. If that repository
    advances between the two downloads a cold boot resolves the weights to commit A, then the
    tokenizer fetch re-resolves the moving ref and lands B -- and afterwards ``refs/main`` holds
    only B. Reading that file for both roles records the weights as B, a commit they never came
    from, and nothing downstream can tell it from correct provenance: the publication gate only
    checks that a commit is PRESENT, and the cross-bucket check only compares identities that are
    wrong in the same way.

    A snapshot directory contains exactly the files its own download fetched, so it answers the
    per-role question the shared ref cannot. This reproduces the split directly.
    """
    from flash.serving.bench import probe as probe_module

    older = "1111111111111111111111111111111111111111"
    newer = "2222222222222222222222222222222222222222"

    with tempfile.TemporaryDirectory() as cache:
        repos = {
            role: entry["repo"]
            for role, entry in probe_module.probe_resolved_revisions("Qwen/Qwen3.6-35B-A3B").items()
        }
        assert repos["model"] == repos["tokenizer"], (
            "the 35B no longer shares one repository across roles, so this race needs rechecking"
        )
        root = Path(cache) / f"models--{repos['model'].replace('/', '--')}"

        # The weights landed in the OLDER snapshot; the tokenizer fetch then advanced the ref.
        model_snapshot = root / "snapshots" / older
        model_snapshot.mkdir(parents=True)
        (model_snapshot / "config.json").write_text("{}")
        # Real weights, not configuration alone: a config-only tree no longer answers for the
        # model role, because every metadata-only fetch creates one.
        (model_snapshot / "model.safetensors").write_text("weights")
        tokenizer_snapshot = root / "snapshots" / newer
        tokenizer_snapshot.mkdir(parents=True)
        (tokenizer_snapshot / "tokenizer.json").write_text("{}")
        refs = root / "refs"
        refs.mkdir(parents=True)
        (refs / "main").write_text(f"{newer}\n")

        with mock.patch("huggingface_hub.constants.HF_HUB_CACHE", cache):
            resolved = probe_module.probe_resolved_revisions("Qwen/Qwen3.6-35B-A3B")

    assert resolved["model"]["commit"] == older, (
        "the weights were recorded at the commit the TOKENIZER download advanced the ref to, so "
        "the published curve names a checkpoint its weights never came from"
    )
    assert resolved["tokenizer"]["commit"] == newer, "the tokenizer lost its own snapshot"
    assert resolved["model"]["source"] == "snapshot-contents"
    assert resolved["model"]["commit"] != resolved["tokenizer"]["commit"], (
        "the split the race produces was collapsed, so it can no longer be seen at all"
    )


def test_ambiguous_and_missing_snapshots_are_reported_rather_than_guessed() -> None:
    """A commit must never be invented, and a genuinely ambiguous cache must say so.

    Re-downloading a repository leaves several snapshots holding the same role, which is ordinary:
    when the ref names one of them it is still the authority. When it does not -- the ref points at
    a tree this role was never downloaded into -- the cache cannot say which snapshot was loaded,
    and reporting a definite-looking commit there would be exactly the false confidence the
    per-role read exists to remove.
    """
    from flash.serving.bench import probe as probe_module

    first = "1111111111111111111111111111111111111111"
    second = "2222222222222222222222222222222222222222"
    elsewhere = "3333333333333333333333333333333333333333"
    repo = probe_module.probe_resolved_revisions("Qwen/Qwen3.5-9B")["model"]["repo"]

    def _cache_with(ref_commit: str) -> Any:
        cache = tempfile.mkdtemp()
        root = Path(cache) / f"models--{repo.replace('/', '--')}"
        for commit in (first, second):
            snapshot = root / "snapshots" / commit
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            # Weights, so both trees are genuine model candidates; ambiguity between two
            # WEIGHT-bearing snapshots is what this test is about.
            (snapshot / "model.safetensors").write_text("weights")
        (root / "refs").mkdir(parents=True)
        (root / "refs" / "main").write_text(f"{ref_commit}\n")
        return cache

    # The ref names one of the snapshots holding the role: ordinary, and the ref is authoritative.
    with mock.patch("huggingface_hub.constants.HF_HUB_CACHE", _cache_with(second)):
        commit, source = probe_module._local_snapshot_commit(repo, None, "model")
    assert (commit, source) == (second, "local-cache-ref")

    # The ref points where this role was never downloaded: the ambiguity must be visible.
    with mock.patch("huggingface_hub.constants.HF_HUB_CACHE", _cache_with(elsewhere)):
        commit, source = probe_module._local_snapshot_commit(repo, None, "model")
    assert source == "snapshot-contents-ambiguous", (
        "an unresolvable cache reported a commit as though it were certain"
    )

    # And the publication gate must refuse it. A guess that reaches the artifact reads exactly like
    # a resolved commit, so presence alone is not the bar.
    namespace = _bench_namespace("_require_resolved_checkpoint")
    ambiguous = {
        "resolved_revisions": {
            "model": {"repo": repo, "commit": first, "source": "snapshot-contents-ambiguous"},
            "tokenizer": {"repo": repo, "commit": first, "source": "snapshot-contents"},
        }
    }
    with pytest.raises(RuntimeError, match="could not be resolved to one snapshot"):
        namespace["_require_resolved_checkpoint"](ambiguous, "canary")

    # A pin that is already a hash needs no cache at all, and an empty cache invents nothing.
    with (
        tempfile.TemporaryDirectory() as empty,
        mock.patch("huggingface_hub.constants.HF_HUB_CACHE", empty),
    ):
        assert probe_module._local_snapshot_commit(repo, None, "model") == (None, None)
        pinned = "0123456789abcdef0123456789abcdef01234567"
        assert probe_module._local_snapshot_commit(repo, pinned, "model") == (
            pinned,
            "pinned-hash",
        )


def test_provenance_records_a_resolved_commit_for_unpinned_repositories() -> None:
    """Two of three hosted models pin nothing, so the run must record what it actually loaded.

    Resolution reads the DOWNLOADED snapshot, never the Hub. Asking the Hub returns what the
    repository points at now; for an unpinned repo that advanced between engine init and probe that
    is false provenance, which is worse than none because it looks authoritative. A failure to
    resolve must be recorded with its reason, never invented.
    """
    from flash.serving.bench import probe as probe_module

    assert "resolved_revisions" in inspect.getsource(probe_module.probe_all), (
        "probe_all does not record resolved revisions"
    )
    source = inspect.getsource(probe_module.probe_resolved_revisions) + inspect.getsource(
        probe_module._local_snapshot_commit
    )
    # A Hub lookup reports the repository's CURRENT head, not the commit this container loaded.
    for hub_call in ("HfApi", "model_info"):
        assert hub_call not in source, f"provenance asks the Hub via {hub_call}"

    sha = "0123456789abcdef0123456789abcdef01234567"

    with tempfile.TemporaryDirectory() as cache:
        repos = {
            entry["repo"]
            for entry in probe_module.probe_resolved_revisions("Qwen/Qwen3.5-9B").values()
        }
        for repo in repos:
            refs = Path(cache) / f"models--{repo.replace('/', '--')}" / "refs"
            refs.mkdir(parents=True, exist_ok=True)
            (refs / "main").write_text(f"{sha}\n")

        with mock.patch("huggingface_hub.constants.HF_HUB_CACHE", cache):
            resolved = probe_module.probe_resolved_revisions("Qwen/Qwen3.5-9B")

    assert set(resolved) == {"model", "tokenizer", "processor"}
    for role, entry in resolved.items():
        assert entry["commit"] == sha, f"{role} did not record the snapshot it loaded"
        assert entry["source"] == "local-cache-ref", f"{role} lost how the commit was determined"
        assert entry["repo"], f"{role} did not record its repository"
        # `pinned` distinguishes a production guarantee from an observation; the 9B pins nothing.
        assert entry["pinned"] is None

    # A revision that is ALREADY a hash needs no cache lookup: the pin is the commit, so an empty
    # cache must still resolve it rather than degrading.
    commit, how = probe_module._local_snapshot_commit("Qwen/Qwen3.5-9B", sha)
    assert (commit, how) == (sha, "pinned-hash")

    # No snapshot degrades to a recorded reason rather than killing a lane that already paid, and
    # never to a fabricated hash.
    with (
        tempfile.TemporaryDirectory() as empty,
        mock.patch("huggingface_hub.constants.HF_HUB_CACHE", empty),
    ):
        degraded = probe_module.probe_resolved_revisions("Qwen/Qwen3.5-9B")
    for role, entry in degraded.items():
        assert entry["commit"] is None, f"{role} invented a commit"
        assert entry["reason"], f"{role} lost the failure reason"


def test_the_fitting_bound_is_enforced_against_a_call_holding_the_gil() -> None:
    """The bound must be enforced by a mechanism that does NOT need the GIL to run.

    This is the failure mode a `threading.Timer` could not cover, and it was verified by direct
    measurement rather than reasoning: a timer armed at 1s did not fire during a 60s GIL-holding C
    call, because its callback is PYTHON and must first acquire the very GIL the stalled call is
    holding. `faulthandler`'s watchdog runs in C and calls `_exit` without touching the GIL, so it
    fires on schedule.

    That distinction is the whole point of the watchdog: `fit_prompt_to_tokens` calls a Rust
    tokenizer extension, so the exact call this guards is the one that holds the GIL while the
    container keeps billing. Asserted end-to-end in a subprocess -- an in-process test cannot
    observe its own termination, and a source-grep would keep passing if the mechanism silently
    stopped working.
    """
    program = textwrap.dedent(
        """
        import sys
        from flash.serving.bench import driver

        # Shrink the grace so the test costs ~1s instead of ~30. The MECHANISM under test is
        # unchanged; only the deadline moves.
        driver._FITTING_WATCHDOG_GRACE_SECONDS = 0.5
        driver._arm_fitting_watchdog(0.5, label="gil-probe")
        # A single C call holding the GIL for far longer than the deadline. A Python-level timer
        # cannot preempt this; the watchdog must.
        sum(range(1, 6_000_000_000))
        # Only reachable if the watchdog never fired.
        sys.exit(0)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(REPO_ROOT),
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode != 0, (
        "the fitting watchdog did not end a process blocked in a GIL-holding C call, so a stuck "
        "tokenizer outlives its bound while the container keeps billing"
    )
    assert b"gil-probe" in completed.stdout + completed.stderr, (
        "the watchdog terminated without naming what it was guarding, leaving an unexplained exit"
    )


def test_the_fitting_watchdog_disarms_and_supersedes_cleanly() -> None:
    """A watchdog must not outlive the fitting it guards, nor leave an earlier deadline running.

    Both directions kill a HEALTHY container: a watchdog left armed after its loop finishes fires
    against work that already succeeded, and a superseded one fires against a bound that is no
    longer being enforced. `faulthandler` keeps a single process-wide deadline, so re-arming must
    replace rather than stack.
    """
    from flash.serving.bench import driver as driver_module

    # The grace must outlast the bound it guards, or a final fit that legitimately starts just
    # inside the bound is killed while doing exactly what it was reserved to do.
    assert driver_module._FITTING_WATCHDOG_GRACE_SECONDS > 0.0

    assert not driver_module._fitting_watchdog_armed, "a watchdog leaked from an earlier test"
    driver_module._arm_fitting_watchdog(600.0, label="probe")
    assert driver_module._fitting_watchdog_armed, "no watchdog was armed"
    driver_module._disarm_fitting_watchdog()
    assert not driver_module._fitting_watchdog_armed, "the watchdog outlives the fitting it guards"

    # Re-arming, then disarming ONCE, must leave nothing running -- otherwise a stacked deadline
    # from the first arm would still terminate a healthy container.
    with driver_module.fitting_watchdog(600.0, label="outer"):
        driver_module._arm_fitting_watchdog(600.0, label="inner")
    assert not driver_module._fitting_watchdog_armed, "a superseded watchdog is still armed"


def test_the_pool_arms_the_watchdog_around_its_whole_fitting_loop() -> None:
    """The bound and the watchdog must come from the SAME expression.

    A watchdog armed at a hardcoded number would drift from the bound the ledger reserved, and the
    lane would either die inside its funded time or bill past it.
    """
    from flash.serving.bench import driver as driver_module

    source = inspect.getsource(driver_module._build_prompt_pool)
    assert "fitting_watchdog(bound" in source, "the pool does not arm the watchdog at its bound"
    # The scope IS the release: `fitting_watchdog` disarms in a `finally`, so it covers the raising
    # path a trailing disarm statement misses. See
    # `test_the_fitting_watchdog_is_released_when_a_fit_raises`.
    assert "with fitting_watchdog(" in source, (
        "the pool leaves the watchdog armed after fitting, so it can kill a container mid-measurement"
    )
    tree = ast.parse(textwrap.dedent(source))
    assigns = {
        target.id: ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "prompt_fit_seconds_bound" in assigns.get("bound", ""), (
        "the bound is not derived from the reserved fitting bound"
    )
    assert "bound" in assigns.get("deadline", ""), "the loop deadline is not the armed bound"


def test_warmups_reserve_and_bound_the_fit_that_precedes_each_request() -> None:
    """Every warmup fits a prompt before it sends one, on the rented container.

    The fit happens outside `run_request`, so pricing a warmup at `REQUEST_TIMEOUT_SECONDS` alone
    left a phase that is enforced but unfunded -- and unguarded, since the pool's watchdog covers
    only the sweep's own pools.
    """
    source = BENCH_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
    }
    # The warmup itself lives in the driver now; the estimators that must FUND it stay in the
    # script, so this guard reads one function from each module.
    warmup_node = ast.parse(textwrap.dedent(inspect.getsource(run_warmup))).body[0]
    withs = [
        ast.unparse(item.context_expr)
        for node in ast.walk(warmup_node)
        if isinstance(node, ast.With | ast.AsyncWith)
        for item in node.items
    ]
    assert any("fitting_watchdog" in expr for expr in withs), (
        "a warmup fit is unguarded, so a stuck tokenizer call bills to the class method timeout"
    )
    assert any("warmup_fit_seconds_bound" in expr for expr in withs), (
        "the warmup guard does not use the bound the estimators reserve"
    )

    calls = [ast.unparse(node) for node in ast.walk(warmup_node) if isinstance(node, ast.Call)]
    assert any("fit_prompt_to_tokens" in call for call in calls)

    # Both estimators must price the fit, not just the request.
    for name in ("_canary_gpu_seconds_estimate", "_sweep_gpu_seconds_estimate"):
        body = ast.unparse(nodes[name])
        assert "_FUNDED_WARMUP_FIT_SECONDS" in body, (
            f"{name} funds warmup requests but not their fits"
        )


def test_the_drain_reap_is_bounded_and_refuses_to_measure_around_a_live_request() -> None:
    """`Task.cancel()` only REQUESTS cancellation.

    If the engine's async-generator close is blocked in a backend call, an unbounded `await task`
    after the cancel sits until the class-wide method timeout -- losing the whole bucket artifact the
    drain exists to preserve. And a request still running into the next cell contaminates the
    measurement that follows it, so a task that will not die must end the container rather than be
    measured around.
    """
    from flash.serving.bench import driver as driver_module

    source = inspect.getsource(driver_module._drain)
    assert "asyncio.wait_for" in source, "the reap is unbounded and can reach the method timeout"
    assert "_DRAIN_REAP_SECONDS" in source, "the reap bound is not the declared one"
    assert "os._exit" in source, (
        "an uncancellable request is measured around, contaminating the next cell"
    )
    # By the time the reap runs the request has already exceeded its own timeout and been
    # cancelled, so this covers cleanup, not work, and must stay small next to the drain allowance.
    assert 0.0 < driver_module._DRAIN_REAP_SECONDS < driver_module.REQUEST_TIMEOUT_SECONDS

    tree = ast.parse(textwrap.dedent(source))
    waits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("wait_for")
    ]
    assert waits, "no bounded wait in the reap"
    for wait in waits:
        assert any(kw.arg == "timeout" for kw in wait.keywords), "the reap wait has no timeout"


def test_a_cell_stops_only_after_a_conclusive_run_of_failures() -> None:
    """A cell whose requests all fail measures nothing, but overload is itself the measurement.

    So the stop needs an attempt floor: an error rate under saturation is the number the envelope is
    for, and cutting that cell short would delete it. Zero successes across a large sample is a
    different thing -- a broken engine billing for a window that cannot produce a curve point.
    """
    from flash.serving.bench import driver as driver_module

    def _record(error: str | None) -> RequestRecord:
        return RequestRecord(
            uid="r",
            base_model="m",
            bucket="b",
            concurrency=4,
            block=0,
            started_at=0.0,
            ok=error is None,
            error=error,
        )

    failed = [_record("boom") for _ in range(driver_module._CONCLUSIVE_FAILURE_ATTEMPTS)]
    assert driver_module._cell_is_conclusively_failed(failed)

    # One success anywhere in the sample means the engine is serving; the failures are data.
    mixed = list(failed)
    mixed[-1] = _record(None)
    assert not driver_module._cell_is_conclusively_failed(mixed)
    mixed = list(failed)
    mixed[0] = _record(None)
    assert not driver_module._cell_is_conclusively_failed(mixed)

    # Below the floor a burst of failures is not yet conclusive.
    assert not driver_module._cell_is_conclusively_failed(
        failed[: driver_module._CONCLUSIVE_FAILURE_ATTEMPTS - 1]
    )
    assert not driver_module._cell_is_conclusively_failed([])
    assert driver_module._CONCLUSIVE_FAILURE_ATTEMPTS >= 32, (
        "the floor is low enough that a merely-overloaded cell is cut short"
    )

    assert "_cell_is_conclusively_failed" in inspect.getsource(driver_module.run_cell), (
        "run_cell never consults the stop, so a dead engine bills its whole window"
    )


def test_every_lane_settles_the_scaledown_tail_it_keeps_billing() -> None:
    """`min_containers=0` does not release the GPU at the last return.

    The container lives out its `scaledown_window` still allocated and still billing. A settlement
    REPLACES the conservative reservation, so omitting that tail silently under-reports every lane
    in the one direction a budget must never err -- and the ledger is the only record of the spend.
    """
    namespace = _bench_namespace("_billable_lane_seconds", "SCALEDOWN_WINDOW_SECONDS")
    window = float(namespace["SCALEDOWN_WINDOW_SECONDS"])
    assert window > 0.0
    assert namespace["_billable_lane_seconds"](100.0) == pytest.approx(100.0 + window)
    # Zero measured wall still owes the tail: a lane that dies immediately after its first call
    # still leaves a container allocated.
    assert namespace["_billable_lane_seconds"](0.0) == pytest.approx(window)

    # Every settle site must go through it, including the failure path -- a lane that dies mid-sweep
    # has already spent, and that is exactly when accurate accounting matters most.
    tree = ast.parse(BENCH_APP.read_text(encoding="utf-8"))
    lane_settles = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "lane.settle"
    ]
    assert len(lane_settles) >= 3, "expected a settle on the canary, sweep and failure paths"
    # Exactly ONE site reaches the ledger, and it applies the tail. Centralizing it is what makes
    # the tail unforgettable: a lane cannot settle without going through this call.
    ledger_settles = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("ledger.settle")
    ]
    assert len(ledger_settles) == 1, (
        f"expected one ledger settle site carrying the tail; found {len(ledger_settles)}"
    )
    for settle in ledger_settles:
        assert any("_billable_lane_seconds" in ast.unparse(arg) for arg in settle.args), (
            f"a settle site reports raw call wall and under-reports its lane: {ast.unparse(settle)}"
        )


def test_the_execution_digest_covers_the_functions_that_schedule_the_work() -> None:
    """A digest over prompt and metric code alone cannot notice a scheduling change.

    Replacement policy, window cutoff and drain handling all move attempts, rates and latency while
    every prompt and metric function stays byte-identical. Two artifacts would then carry the same
    digest and be compared as if produced by the same program.
    """
    from flash.serving.bench import driver as driver_module
    from flash.serving.bench import workload as workload_module

    for name in ("run_cell", "run_request", "_drain"):
        assert name in workload_module._DRIVER_SOURCES, (
            f"{name} schedules measured work but changing it does not move the digest"
        )
        assert hasattr(driver_module, name), f"_DRIVER_SOURCES names {name}, which no longer exists"

    before = workload_module._execution_digest()
    original = driver_module.run_cell
    try:
        # A scheduling change with no prompt or metric change must still move the digest.
        def run_cell(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - never called
            """A different scheduler."""
            return original(*args, **kwargs)

        driver_module.run_cell = run_cell
        assert workload_module._execution_digest() != before, (
            "rewriting the cell scheduler leaves the digest unchanged"
        )
    finally:
        driver_module.run_cell = original
    assert workload_module._execution_digest() == before, (
        "the digest is not stable across a restore"
    )


def test_provenance_is_captured_before_any_probe_that_could_rewrite_it() -> None:
    """The revision must be read before anything that can advance the ref it reads.

    `_local_snapshot_commit` reports an unpinned model's commit from the HF cache's `refs/main`,
    which is mutable. A network-capable checkpoint load re-resolves that ref and rewrites it to
    whatever the Hub holds now, so a probe running first could advance the very file the revision
    probe reads -- and the artifact would attribute a measured number to a commit the engine never
    loaded.

    Two independent guarantees, asserted separately because either alone can be lost in a refactor:
    the config read is pinned to the local cache, and revisions are captured first regardless.
    """
    import ast
    import inspect

    from flash.serving.bench import probe as probe_module

    source = inspect.getsource(probe_module._gdn_config_values)
    call = next(
        node
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
    )
    pinned = [kw for kw in call.keywords if kw.arg == "local_files_only"]
    assert pinned, "the config load may reach the Hub and rewrite the ref the revision probe reads"
    assert pinned[0].value.value is True, (
        "local_files_only is passed but not enabled, so the load can still reach the Hub"
    )

    body = ast.parse(textwrap.dedent(inspect.getsource(probe_module.probe_all))).body[0]
    keys: list[str] = []
    for node in ast.walk(body):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            break
    assert keys[:1] == ["resolved_revisions"], (
        f"provenance is not captured first; probe_all builds {keys}"
    )


def test_the_execution_digest_covers_the_abandonment_policy() -> None:
    """Which cells EXIST in a published curve is a measurement decision, not an implementation one.

    `_cell_is_conclusively_failed` decides when a bucket sweep stops opening concurrency points, so
    loosening it turns a halted sweep into a longer one: more points, a different ceiling and a
    different knee, while every prompt, every request and every metric function stays byte-identical.

    Its threshold needs separate cover from its source. The function's body names
    `_CONCLUSIVE_FAILURE_ATTEMPTS` but never reveals the value, so retuning the constant alone leaves
    `inspect.getsource` byte-identical -- the digest would not move for a change that alters which
    cells were measured.
    """
    from flash.serving.bench import driver as driver_module
    from flash.serving.bench import workload as workload_module

    assert "_cell_is_conclusively_failed" in workload_module._DRIVER_SOURCES, (
        "the abandonment policy decides which cells exist but does not move the digest"
    )
    assert hasattr(driver_module, "_cell_is_conclusively_failed"), (
        "_DRIVER_SOURCES names a function that no longer exists"
    )
    assert "_CONCLUSIVE_FAILURE_ATTEMPTS" in workload_module._DRIVER_CONSTANTS, (
        "retuning the abandonment threshold leaves the digest unchanged"
    )

    before = workload_module._execution_digest()
    original_threshold = driver_module._CONCLUSIVE_FAILURE_ATTEMPTS
    try:
        # A threshold retune with no source change anywhere must still move the digest.
        driver_module._CONCLUSIVE_FAILURE_ATTEMPTS = original_threshold * 2
        assert workload_module._execution_digest() != before, (
            "retuning how many attempts abandon a grid leaves the digest unchanged"
        )
    finally:
        driver_module._CONCLUSIVE_FAILURE_ATTEMPTS = original_threshold

    original = driver_module._cell_is_conclusively_failed
    try:

        def _cell_is_conclusively_failed(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            """A different abandonment rule."""
            return original(*args, **kwargs)

        driver_module._cell_is_conclusively_failed = _cell_is_conclusively_failed
        assert workload_module._execution_digest() != before, (
            "rewriting the abandonment rule leaves the digest unchanged"
        )
    finally:
        driver_module._cell_is_conclusively_failed = original
    assert workload_module._execution_digest() == before, (
        "the digest is not stable across a restore"
    )


def test_the_fitting_watchdog_is_released_when_a_fit_raises() -> None:
    """A raised fit must not leave a process-wide deadline armed on a retained container.

    `fit_prompt_to_tokens` raising is ordinary: a tokenizer that cannot reach the bucket's target
    length says so. The watchdog it runs under is `faulthandler`'s PROCESS-wide deadline, and the
    Modal container outlives the call, so an arm that leaks past the raise stays live and ends some
    later, unrelated call on that container -- the next probe, warmup or bucket -- at a bound that
    belonged to a lane which already returned. Enforced structurally, since a trailing disarm
    statement is exactly what an exception skips.
    """
    from flash.serving.bench import driver as driver_module

    source = textwrap.dedent(inspect.getsource(driver_module._build_prompt_pool))
    assert "with fitting_watchdog(" in source, (
        "the pool arms its watchdog without scoping it, so a raised fit leaks the deadline"
    )
    assert "_arm_fitting_watchdog(" not in source, (
        "the pool still arms the watchdog directly, bypassing the scope that releases it"
    )

    # Behavioural, not just structural: a raising fit must leave nothing armed.
    armed: list[float] = []
    original_fit = driver_module.fit_prompt_to_tokens
    original_dump = driver_module.faulthandler.dump_traceback_later
    original_cancel = driver_module.faulthandler.cancel_dump_traceback_later

    def exploding_fit(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("cannot fit this bucket")

    def record_dump(timeout: float, **kwargs: Any) -> None:
        armed.append(timeout)

    def record_cancel() -> None:
        armed.clear()

    try:
        driver_module.fit_prompt_to_tokens = exploding_fit
        driver_module.faulthandler.dump_traceback_later = record_dump
        driver_module.faulthandler.cancel_dump_traceback_later = record_cancel
        with pytest.raises(RuntimeError, match="cannot fit"):
            driver_module._build_prompt_pool(
                object(),
                BUCKETS[0],
                concurrency=1,
                block=0,
                min_requests=1,
            )
    finally:
        driver_module.fit_prompt_to_tokens = original_fit
        driver_module.faulthandler.dump_traceback_later = original_dump
        driver_module.faulthandler.cancel_dump_traceback_later = original_cancel
        driver_module._fitting_watchdog_armed = False

    assert armed == [], (
        "a raised fit left the process-wide fitting deadline armed on a retained container"
    )


def test_every_request_sends_the_preregistered_choice_count() -> None:
    """`n` must be SENT, not inherited from the serving default the checksum does not cover.

    `workload_checksum()` records `n=N_CHOICES`. Omitting the field from the payload made that claim
    rest on `OpenAIGenerateRequest.n`, a separate default in production code. The two are equal
    today and independent tomorrow: raising the production default would make every request generate
    multiple choices -- different generation work, different token usage, different latency -- under
    a checksum still asserting one choice, so the artifact could not be distinguished from a valid
    campaign's.
    """
    from flash.serving.bench import driver as driver_module

    payload = driver_module._payload_for(
        "Qwen/Qwen3.5-9B", [{"role": "user", "content": "x"}], 8, "u"
    )
    assert "n" in payload, (
        "the payload omits n, so the measured choice count is the serving default, not the "
        "preregistered one the checksum records"
    )
    assert payload["n"] == N_CHOICES, (
        f"the payload sends n={payload['n']} but the checksum records n={N_CHOICES}"
    )


def test_the_execution_digest_covers_the_request_spawner() -> None:
    """Retuning what each request is issued with must move the digest.

    `_make_spawner` decides when the measurement clock starts and passes the bucket output limit,
    the fitted length, the bucket target and the invocation identity into every request. Changing
    any of them alters records and rates while every source and constant otherwise digested stays
    byte-identical, letting two materially different campaigns publish one checksum.
    """
    from flash.serving.bench import driver as driver_module
    from flash.serving.bench import workload as workload_module

    assert "_make_spawner" in workload_module._DRIVER_SOURCES, (
        "changing what every request is issued with does not move the execution digest"
    )
    assert hasattr(driver_module, "_make_spawner"), (
        "_DRIVER_SOURCES names _make_spawner, which no longer exists"
    )

    before = workload_module._execution_digest()
    original = driver_module._make_spawner
    try:

        def _make_spawner(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - never called
            """A spawner that issues different work."""
            return original(*args, **kwargs)

        driver_module._make_spawner = _make_spawner
        assert workload_module._execution_digest() != before, (
            "rewriting the request spawner leaves the digest unchanged"
        )
    finally:
        driver_module._make_spawner = original
    assert workload_module._execution_digest() == before, (
        "the digest is not stable across a restore"
    )


def test_the_drain_reap_is_bounded_across_the_whole_pending_set() -> None:
    """R16: one reap allowance per CELL, not one per pending task.

    Both estimators fund `drain_reap_seconds()` once per cell. Applying that timeout separately
    inside the loop over pending requests multiplies it by the number of tasks: cancelled engine
    streams that serialize their cleanup let a concurrency-16 cell spend 16 x 30s reaping against a
    30s reservation, billing past the ceiling `BudgetLedger` exists to enforce.

    Driven by a FAKE clock rather than real waits. Tasks that honour cancellation promptly return
    before their timeout, so no wall time passes and a per-task bound looks identical to a shared
    one; the reported defect is specifically about cleanups that DO consume their allowance. The
    clock advances by whatever each wait was granted, which reproduces that serialization exactly
    and makes the requested timeouts -- the thing under test -- deterministic.
    """
    from flash.serving.bench import driver as driver_module

    granted: list[float] = []
    now = [1000.0]

    async def _recording_wait_for(
        awaitable: Any,
        # Mirrors asyncio.wait_for's own signature -- the granted timeout IS what this test records,
        # so the parameter cannot be renamed away.
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> Any:
        granted.append(float(timeout if timeout is not None else 0.0))
        # A cleanup that used its whole allowance, which is the case that multiplies.
        now[0] += float(timeout or 0.0)
        return await awaitable

    async def _finishes() -> Any:
        return None

    async def _exercise() -> None:
        tasks = {asyncio.ensure_future(_finishes()) for _ in range(4)}
        await asyncio.sleep(0)
        # Every task is pending as far as the drain is concerned, so all four enter the reap.
        # `monotonic` must be a lambda, not `return_value=`: the fake clock has to be read at call
        # time so the advance inside `_recording_wait_for` is visible to the next iteration.
        with (
            mock.patch.object(driver_module, "_DRAIN_REAP_SECONDS", 30.0),
            mock.patch.object(driver_module.time, "monotonic", lambda: now[0]),  # noqa: PT008
            mock.patch.object(driver_module.asyncio, "wait_for", _recording_wait_for),
            mock.patch.object(
                driver_module.asyncio, "wait", mock.AsyncMock(return_value=(set(), tasks))
            ),
        ):
            await driver_module._drain(
                tasks,
                base_model="Qwen/Qwen3.5-9B",
                bucket="short_interactive",
                concurrency=4,
                block=0,
                spawned_at={},
                spawned_uid={},
            )

    asyncio.run(_exercise())

    assert granted, "no reap timeout was ever applied"
    assert sum(granted) <= 30.0 + 1e-6, (
        f"the reap was granted {sum(granted):.1f}s against a 30s per-cell reservation: the bound "
        "is being applied per task rather than across the pending set"
    )
    # Not vacuous: the first task must really get the allowance, so this cannot pass by never
    # reaping at all.
    assert granted[0] == pytest.approx(30.0)


def test_the_grid_stop_policy_is_digested() -> None:
    """R16: the predicate that ABANDONS a concurrency grid must move the execution digest.

    `_cell_is_conclusively_failed` stops replacing requests within one cell; the decision that ends
    the grid lived inline in the entrypoint script, outside every digested source. Loosening it
    would change which cells the published curve contains -- and with them the ceiling, knee and
    saturation point -- while `workload_checksum` stayed byte-identical, so two incompatible
    campaigns could claim one checksum.
    """
    from flash.serving.bench import driver as driver_module
    from flash.serving.bench import workload as workload_module

    assert "grid_should_halt" in workload_module._DRIVER_SOURCES
    assert hasattr(driver_module, "grid_should_halt")

    before = workload_module.workload_checksum()
    original = driver_module.grid_should_halt

    def _looser(result: CellResult) -> bool:
        return result.succeeded == 0

    try:
        driver_module.grid_should_halt = _looser  # type: ignore[assignment]
        assert workload_module.workload_checksum() != before, (
            "retuning the grid stop policy left the checksum unchanged"
        )
    finally:
        driver_module.grid_should_halt = original  # type: ignore[assignment]
    assert workload_module.workload_checksum() == before


def test_an_unnamed_checkpoint_refuses_publication() -> None:
    """R16: a curve that cannot name its weights must not be measured, let alone published.

    `probe_resolved_revisions` is fail-soft: it records `commit=None` with a reason rather than
    inventing a hash. Nothing refused on that, so a missing or unreadable cache ref would publish a
    measurement identified only by a mutable repository name -- and two of the three hosted models
    pin nothing, so once that repository advances the artifact can no longer say what it ran on.
    """
    namespace = _bench_namespace("_require_resolved_checkpoint")
    require = namespace["_require_resolved_checkpoint"]

    good = {
        "resolved_revisions": {
            "model": {"repo": "a/b", "commit": "c" * 40},
            "tokenizer": {"repo": "a/b", "commit": "d" * 40},
            # Absent for text-only models, and resolves against the tokenizer repository, so it is
            # deliberately not required.
            "processor": {"repo": "a/b", "commit": None, "reason": "no processor"},
        }
    }
    require(good, "canary")

    for missing in ("model", "tokenizer"):
        broken = {
            "resolved_revisions": {
                role: dict(entry) for role, entry in good["resolved_revisions"].items()
            }
        }
        broken["resolved_revisions"][missing] = {
            "repo": "a/b",
            "commit": None,
            "reason": "no local snapshot found for the loaded repository",
        }
        with pytest.raises(RuntimeError, match="cannot name the weights"):
            require(broken, "canary")
    # An absent block is not a pass either.
    with pytest.raises(RuntimeError, match="cannot name the weights"):
        require({}, "canary")


def test_both_paid_paths_gate_on_a_named_checkpoint() -> None:
    """R16: the gate must be CALLED on the canary and on every bucket, before the grid opens.

    A predicate nothing invokes is exactly the reported defect, so assert the call sites rather
    than only the helper.
    """
    source = BENCH_APP.read_text(encoding="utf-8")
    # The canary keeps its own client-side call: it re-checks what `certify` returned, since the
    # container can only gate against its own `base_model`. The in-container paths reach the same
    # requirement through the shared gate, which is applied on BOTH before anything is warmed.
    assert '_require_resolved_checkpoint(probe, "canary")' in source
    assert "_require_resolved_checkpoint(probe, where)" in source
    assert '_gate_container_provenance(self, provenance, "canary")' in source
    assert '_gate_container_provenance(engine, provenance, f"bucket {bucket_name!r}")' in source


def test_a_replacement_container_is_probed_before_it_is_warmed() -> None:
    """R16: gate the container BEFORE paying five sequential warmups on it.

    A cold replacement whose card or kernel path fails the publication gate was warmed first: each
    warmup can spend its fitting allowance plus the full request timeout, so the harness could burn
    roughly an hour of paid GPU time on a container it was always going to reject.
    """
    tree = ast.parse(BENCH_APP.read_text(encoding="utf-8"))
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_bucket"
    )
    # `gpu_matches` and the two `_require_*` predicates now live inside `_gate_container_provenance`
    # so that both in-container callers apply the identical sequence. What must still hold here is
    # the ordering: probe, gate, and only then pay for warmups.
    watched = {
        "_ensure_warm",
        "_probe_in_container_within_bound",
        "_gate_container_provenance",
    }
    order = [
        (node.func.id, node.lineno)
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in watched
    ]
    positions = dict(order)
    assert "_ensure_warm" in positions, "the bucket no longer warms a cold replacement"
    for gate in (
        "_probe_in_container_within_bound",
        "_gate_container_provenance",
    ):
        assert gate in positions, f"{gate} is not applied by the bucket"
        assert positions[gate] < positions["_ensure_warm"], (
            f"{gate} runs AFTER the warmup, so a rejectable container is paid for before it is "
            "refused"
        )


def test_the_degraded_verdict_is_digested() -> None:
    """R17: the predicate that classifies a cell as FAILED must move the execution digest.

    `summarize_curve` is digested, but it delegates to `CellResult.degraded` to choose which cells
    are usable and where saturation begins. The digest resolves `getattr(metrics, name)`, which
    reaches module-level objects only, so the property's body sat outside every digested source:
    dropping its zero-in-window-success test would move the ceiling, knee and saturation point of
    every published curve while `workload_checksum` stayed byte-identical.
    """
    from flash.serving.bench import metrics as metrics_module
    from flash.serving.bench import workload as workload_module

    assert ("CellResult", "degraded") in workload_module._METRIC_PROPERTIES

    before = workload_module.workload_checksum()
    original = metrics_module.CellResult.degraded

    def _looser(self: CellResult) -> bool:
        # Drops the `succeeded_in_window` test -- the exact retune that would silently republish a
        # zero-throughput cell as usable.
        return self.error_rate > self.max_error_rate

    try:
        metrics_module.CellResult.degraded = property(_looser)  # type: ignore[assignment]
        assert workload_module.workload_checksum() != before, (
            "retuning the degraded verdict left the checksum unchanged"
        )
    finally:
        metrics_module.CellResult.degraded = original  # type: ignore[assignment]
    assert workload_module.workload_checksum() == before


def test_the_execution_digest_covers_the_latency_and_ttft_arithmetic() -> None:
    """R21: the getters that turn timestamps into durations must move the execution digest.

    `reduce_cell` is digested, but its source only NAMES `record.ttft` and `record.latency` -- the
    subtraction that defines each metric lives in the property bodies, which `getattr(metrics, name)`
    cannot reach. Redefining either would move every published TTFT and latency percentile while
    every digested source stayed byte-identical, so two campaigns run under materially different
    metric contracts would compare as though they shared one.
    """
    from flash.serving.bench import metrics as metrics_module
    from flash.serving.bench import workload as workload_module

    for prop in ("ttft", "latency"):
        assert ("RequestRecord", prop) in workload_module._METRIC_PROPERTIES

    for prop in ("ttft", "latency"):
        before = workload_module.workload_checksum()
        original = getattr(metrics_module.RequestRecord, prop)

        def _retuned(self: RequestRecord) -> float | None:
            # Measures from FIRST TOKEN rather than from send -- a different metric contract that
            # leaves every digested source byte-identical.
            if self.first_token_at is None or self.finished_at is None:
                return None
            return self.finished_at - self.first_token_at

        try:
            setattr(metrics_module.RequestRecord, prop, property(_retuned))
            assert workload_module.workload_checksum() != before, (
                f"retuning RequestRecord.{prop} left the checksum unchanged"
            )
        finally:
            setattr(metrics_module.RequestRecord, prop, original)
        assert workload_module.workload_checksum() == before


def test_the_gdn_config_is_read_at_the_commit_the_provenance_captured() -> None:
    """R21: `local_files_only` pins the config to the cache, not to a COMMIT.

    An unpinned repository passes `revision=None`, which still resolves through the cache's mutable
    `refs/main`. A tokenizer or processor download can advance that ref during boot -- and the 35B
    model is exactly the case where it can, because its weights and tokenizer share one unpinned
    repository. The resolver would then bind head dims from a newer config while the engine runs
    older weights, and the artifact would publish a GDN backend for a checkpoint nobody served.

    So the config load must resolve against the commit `_local_snapshot_commit` reports for the
    MODEL role, and `config_source` must name that same commit rather than the `None` that was
    asked for.
    """
    from flash.serving.bench import probe as probe_module

    captured = "a" * 40
    with (
        mock.patch.object(probe_module, "_served_checkpoint", return_value=("acme/unpinned", None)),
        mock.patch.object(
            probe_module, "_local_snapshot_commit", return_value=(captured, "snapshot-contents")
        ),
    ):
        repo, revision = probe_module._config_checkpoint("acme/unpinned")

    assert repo == "acme/unpinned"
    assert revision == captured, (
        "the config resolves through the mutable ref instead of the captured commit"
    )

    # A repository that pins its own revision keeps it: the pin IS the commit, and re-deriving it
    # from the cache would make the probe depend on a download that may not have happened.
    with (
        mock.patch.object(
            probe_module, "_served_checkpoint", return_value=("acme/pinned", "b" * 40)
        ),
        mock.patch.object(
            probe_module, "_local_snapshot_commit", return_value=("c" * 40, "snapshot-contents")
        ),
    ):
        assert probe_module._config_checkpoint("acme/pinned") == ("acme/pinned", "b" * 40)


def test_a_warmup_request_is_bounded_against_a_hung_stream_close() -> None:
    """R17: `run_request`'s own timeout is not an upper bound when cancellation cleanup hangs.

    `asyncio.wait_for` cancels the inner coroutine and then WAITS for that cancellation to finish,
    so a stream whose close blocks holds `run_request` open with no `TimeoutError` ever raised. A
    measured cell survives it because `_drain` awaits a shielded task; the warmup awaited directly,
    while both estimators reserve only the request timeout for it.

    Driven IN A SUBPROCESS under a hard wall-clock kill, because the unbounded form of this defect
    hangs rather than fails. An in-process ceiling cannot cover it: every such ceiling is an asyncio
    timer, and the two things that would neuter one here are exactly what the unbounded path does --
    it never raises, so nothing propagates, and any fake clock installed to keep the enforced path
    fast also freezes the event loop's own timer clock (`driver.time` IS the `time` module), so the
    ceiling can never expire. `subprocess.run(timeout=...)` is enforced by the OS and needs neither.

    The bound under test is shrunk so the enforced path returns in ~1s rather than the real
    timeout. Only the deadline moves; the MECHANISM is unchanged.
    """
    program = textwrap.dedent(
        """
        import asyncio, sys
        from flash.serving.bench import driver

        driver.REQUEST_TIMEOUT_SECONDS = 0.5
        driver._DRAIN_REAP_SECONDS = 0.5

        async def _hangs(*a, **k):
            # Never completes, and ignores cancellation the way a blocked `aclose()` does.
            await asyncio.Event().wait()

        driver.run_request = _hangs
        # The enforced path ends the container rather than billing an unbounded call. Record that
        # instead of dying, so the exit code below distinguishes enforced from unbounded.
        driver.os._exit = lambda code: sys.exit(7)

        async def _main():
            try:
                await driver.run_request_within_bound(None, "m", [], 1, "uid")
            except (TimeoutError, asyncio.CancelledError):
                sys.exit(8)

        asyncio.run(_main())
        sys.exit(0)
        """
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "the warmup awaited a request with no enforceable bound: it never returned, so a "
            "stream whose close hangs would bill to the class-wide method timeout"
        )
    assert completed.returncode in {7, 8}, (
        f"expected the bound to be enforced (exit 7 or 8), got {completed.returncode}: "
        f"{completed.stdout}{completed.stderr}"
    )


def test_a_request_that_delivers_during_the_reap_is_returned_not_discarded() -> None:
    """A bounded cancellation must not throw away an outcome the container already paid for.

    Cancellation races delivery. `run_request` catches its own failures and RETURNS a failed record
    instead of raising, so a request that finishes between the bound expiring and the cancel landing
    leaves a real `RequestRecord` in the task -- outcome decided, seconds already billed. Raising
    over it fails the lane for a request that actually completed, and the warmup is the gate every
    sweep runs behind, so one such race would refuse a healthy container.

    Driven in a subprocess for the same reason as the bound test above: the fake clock a fast
    in-process version needs would freeze the event loop's own timer.
    """
    program = textwrap.dedent(
        """
        import asyncio, sys
        from flash.serving.bench import driver

        # The enforced bound is the SUM of these two, so the request below must outlive 1.1s to
        # reach the cancellation path at all -- a sleep shorter than the sum returns through the
        # normal path and exercises nothing.
        driver.REQUEST_TIMEOUT_SECONDS = 0.1
        driver._DRAIN_REAP_SECONDS = 1.0

        DELIVERED = {"ok": False, "marker": "delivered-during-the-reap"}

        async def _slow(*a, **k):
            # Outlives the 1.1s bound, then delivers inside the reap -- exactly the race. Swallowing
            # the cancellation and returning is what `run_request` does: it converts its own
            # failures into a failed record rather than propagating.
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                pass
            return DELIVERED

        driver.run_request = _slow
        driver.os._exit = lambda code: sys.exit(7)

        async def _main():
            try:
                got = await driver.run_request_within_bound(None, "m", [], 1, "uid")
            except BaseException as exc:
                print(f"raised {type(exc).__name__}", flush=True)
                sys.exit(8)
            print(f"returned {got!r}", flush=True)
            sys.exit(0 if got is DELIVERED else 9)

        asyncio.run(_main())
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, (
        "a request that delivered during the bounded reap was discarded rather than returned; "
        f"exit {completed.returncode}: {completed.stdout}{completed.stderr}"
    )


def test_the_warmup_reservation_funds_the_enforced_bound() -> None:
    """R17: reserving the nominal request timeout under-funds a path the code itself permits."""
    from flash.serving.bench import driver as driver_module

    assert driver_module.run_request_bound_seconds() > driver_module.REQUEST_TIMEOUT_SECONDS, (
        "the enforced bound collapsed to the request timeout, so the reap is funded by nothing"
    )
    source = BENCH_APP.read_text(encoding="utf-8")
    assert "REQUEST_TIMEOUT_SECONDS + _FUNDED_WARMUP_FIT_SECONDS" not in source, (
        "a warmup is still priced at the nominal request timeout it is no longer bounded by"
    )
    # Four sites: the two per-call priced bounds (`bucket_call_priced_seconds`,
    # `certify_call_priced_seconds`) that each call is now ENFORCED at, plus the canary and sweep
    # estimators that reserve against them. The bound and the reservation must price a warmup
    # identically or a call is authorized to spend more than it was funded for.
    assert source.count("run_request_bound_seconds() + _FUNDED_WARMUP_FIT_SECONDS") == 4, (
        "not every warmup reservation site funds the enforced bound"
    )


def test_curves_from_different_checkpoints_are_not_published_as_one_envelope() -> None:
    """R17: each bucket gates its OWN container; nothing compared those containers to each other.

    A Hub repo that advances mid-sweep, or a replacement container, leaves bucket A measured on one
    commit and bucket B on another. Every payload passes its own non-null check, and the summary
    then fuses their curves into a single ceiling, knee and saturation point describing no model
    that exists.
    """
    namespace = _bench_namespace("_checkpoint_identity", "_require_one_identity")
    gate = namespace["_require_one_identity"]
    checkpoint_identity = namespace["_checkpoint_identity"]

    def require(payloads: list[tuple[str, dict[str, Any]]]) -> dict[str, str]:
        return gate(payloads, checkpoint_identity, "checkpoint")

    def _probe(model: str, tokenizer: str) -> dict[str, Any]:
        return {
            "resolved_revisions": {
                "model": {"commit": model, "repo": "Qwen/Qwen3.5-9B"},
                "tokenizer": {"commit": tokenizer, "repo": "Qwen/Qwen3.5-9B"},
            }
        }

    # Same checkpoint everywhere: publishes.
    require([("canary", _probe("aaa", "bbb")), ("bucket 'short'", _probe("aaa", "bbb"))])

    # A processor role present on one side only must NOT split an otherwise identical identity:
    # it resolves against the tokenizer repo and is absent for text-only models.
    with_processor = _probe("aaa", "bbb")
    with_processor["resolved_revisions"]["processor"] = {"commit": "ccc", "repo": "x"}
    require([("canary", _probe("aaa", "bbb")), ("bucket 'short'", with_processor)])

    for drifted in (_probe("zzz", "bbb"), _probe("aaa", "zzz")):
        with pytest.raises(RuntimeError, match="different checkpoints"):
            require([("canary", _probe("aaa", "bbb")), ("bucket 'short'", drifted)])


def test_the_lane_gates_checkpoint_identity_before_it_summarizes() -> None:
    """R17: the drift gate is worthless if the summary is written before it runs."""
    tree = ast.parse(BENCH_APP.read_text(encoding="utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name == "_run_sweep_lane"
    )
    calls = [
        (ast.unparse(node.func), node.lineno)
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) in {"_require_one_identity", "lane.write"}
    ]
    gate = [line for name, line in calls if name == "_require_one_identity"]
    assert gate, "the lane never compares checkpoints across buckets"
    writes = [line for name, line in calls if name == "lane.write"]
    assert max(writes) > gate[0], "the drift gate runs after every artifact is already written"


def test_a_container_that_fails_its_gates_is_refused_before_it_is_warmed() -> None:
    """`certify` must reject on the probe, not after paying for five warmups.

    The gates used to live only in `_run_canary`, which reads what `certify` returns -- so `certify`
    probed, then warmed unconditionally, and only afterwards did anything look at the probe it was
    already holding. A container on the wrong card or with an unresolved kernel path therefore ran
    CANARY_WARMUP_REQUESTS sequential warmups first, each able to spend its fitting allowance plus
    the full request bound, before being refused for a reason known before the first one started.

    Ordering is the whole property, so this asserts on ordering: the warmup must not have been
    entered at all, which is stronger than the refusal merely happening.
    """
    namespace = _bench_namespace(
        "_gate_container_provenance",
        "_require_resolved_gdn_backend",
        "_require_resolved_checkpoint",
    )
    gate = namespace["_gate_container_provenance"]
    model = BENCH_MODELS[0]
    engine = types.SimpleNamespace(base_model=model)
    right_card = bench_gpu_for(model)

    def _probe(gpu_name: str, *, gdn: bool = True, commit: str = "a" * 40) -> dict[str, Any]:
        return {
            "gpu": {"name": gpu_name},
            "gdn_prefill": {"resolved": "flashinfer" if gdn else None, "reason": "unavailable"},
            "resolved_revisions": {
                "model": {"repo": model, "commit": commit, "source": "snapshot-contents"},
                "tokenizer": {"repo": model, "commit": commit, "source": "snapshot-contents"},
            },
        }

    # A healthy probe passes, so the gate is not simply refusing everything.
    gate(engine, _probe(right_card), "canary")

    # Each of the three rejections a container can earn.
    with pytest.raises(RuntimeError, match="refusing to attribute a measurement to the wrong card"):
        gate(engine, _probe("NVIDIA GeForce RTX 4090"), "canary")
    with pytest.raises(RuntimeError, match="GDN prefill backend is unresolved"):
        gate(engine, _probe(right_card, gdn=False), "canary")
    with pytest.raises(RuntimeError, match="resolved to no commit"):
        gate(engine, _probe(right_card, commit=""), "canary")

    # And `certify` must apply it BETWEEN the probe and the warmup. Read from the real source: the
    # method is built inside `_build_bench_engine`, so it cannot be lifted and executed on its own.
    source = BENCH_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    # `certify` delegates its body to `_certify` so the whole call can be held to its priced
    # bound; the probe/gate/warm sequence this asserts lives there.
    certify = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_certify"
    )
    order = [
        name.func.id if isinstance(name.func, ast.Name) else getattr(name.func, "attr", "")
        for name in ast.walk(certify)
        if isinstance(name, ast.Call)
    ]
    assert "_probe_in_container_within_bound" in order, "certify no longer probes its container"
    assert "_gate_container_provenance" in order, "certify warms a container it never gated"
    assert "run_warmup" in order, "certify no longer warms"
    body = ast.get_source_segment(source, certify) or ""
    gate_at = body.index("_gate_container_provenance")
    warm_at = body.index("run_warmup")
    probe_at = body.index("_probe_in_container_within_bound")
    assert probe_at < gate_at < warm_at, (
        "certify warms before it gates, so a container already known to be unpublishable still "
        "pays for every warmup"
    )


def test_the_provenance_gates_live_in_one_place_so_a_caller_cannot_skip_them() -> None:
    """Three gates that must travel together, applied by one helper.

    `_run_bucket` and `certify` both refuse a container on card, kernel path and checkpoints. Held
    as three separate call sequences, a third caller -- or an edit to one of the two -- can drop a
    gate silently, which is exactly how `certify` came to hold the probe without applying it.
    """
    source = BENCH_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_gate_container_provenance"
    )
    inner = {
        call.func.id
        for call in ast.walk(helper)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert {"_require_resolved_gdn_backend", "_require_resolved_checkpoint"} <= inner, (
        "the shared gate no longer applies both provenance requirements"
    )

    # Nobody re-implements the sequence. The canary keeps its own `gpu_matches` check against the
    # gpu the CALLER expects -- the container can only check its own `base_model` -- but the two
    # in-container callers must reach the requirements through the shared helper.
    for name in ("_run_bucket",):
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef | ast.FunctionDef) and n.name == name
        )
        calls = {
            c.func.id
            for c in ast.walk(node)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        }
        assert "_gate_container_provenance" in calls, f"{name} does not use the shared gate"
        assert not {"_require_resolved_gdn_backend", "_require_resolved_checkpoint"} & calls, (
            f"{name} re-implements the gate sequence instead of using the shared helper"
        )


def test_a_successful_record_finishes_when_its_final_event_arrived() -> None:
    """R22: generator cleanup after the terminal event must not be charged to the request.

    `run_request` awaits the whole `async for`, so a post-loop clock also includes the async
    generator's close. A slow or blocked `aclose` inflates latency and can push an
    already-delivered response past the in-window cutoff, dropping a real completion out of the
    throughput numerator -- one-directional bias, largest exactly under load.
    """
    import time

    from flash.serving.bench import driver

    origin = time.monotonic()

    class _SlowClose:
        """Delivers a clean stream, then stalls in cleanup exactly like a blocked aclose."""

        def _stream_generate(self, payload, forwarded, _lora, _gid):  # type: ignore[no-untyped-def]
            async def _gen():
                yield {"type": "ready", "engine_replica_id": "r0"}
                yield {"type": "delta", "text": "hi"}
                yield {"type": "choice_finished", "finish_reason": "stop"}
                yield {
                    "type": "final",
                    "prompt_tokens": 512,
                    "completion_tokens": 8,
                    "cached_tokens": 0,
                    "cached_tokens_reported": True,
                }
                await asyncio.sleep(0.35)

            return _gen()

    record = asyncio.run(
        driver.run_request(
            engine=_SlowClose(),
            base_model="acme/model",
            bucket="b",
            concurrency=1,
            block=0,
            uid="u0",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            origin=origin,
            expected_prompt_tokens=512,
            bucket_target_tokens=512,
        )
    )

    assert record.error is None, record.error_detail
    assert record.finished_at is not None
    assert record.finished_at < 0.3, (
        "a successful record is still timestamped after generator cleanup, so post-delivery "
        f"stalling is billed as latency (finished_at={record.finished_at})"
    )


def test_a_failed_record_still_reports_the_wall_it_consumed() -> None:
    """The delivered-final clock is for successes only.

    A timeout or engine error has no delivered terminal event, and the wall those records consumed
    is exactly what they exist to report -- so they must keep the post-drain clock.
    """
    import time

    from flash.serving.bench import driver
    from flash.serving.bench.metrics import ERROR_ENGINE

    origin = time.monotonic()

    class _Broken:
        def _stream_generate(self, payload, forwarded, _lora, _gid):  # type: ignore[no-untyped-def]
            async def _gen():
                yield {"type": "ready"}
                await asyncio.sleep(0.2)
                raise RuntimeError("engine died")

            return _gen()

    record = asyncio.run(
        driver.run_request(
            engine=_Broken(),
            base_model="acme/model",
            bucket="b",
            concurrency=1,
            block=0,
            uid="u1",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            origin=origin,
            expected_prompt_tokens=512,
            bucket_target_tokens=512,
        )
    )

    assert record.error == ERROR_ENGINE
    assert record.finished_at is not None
    assert record.finished_at >= 0.2, (
        "a failed record no longer reports the wall it actually consumed"
    )


def test_a_final_delivered_before_a_failing_cleanup_is_still_a_success() -> None:
    """R24: a fault raised AFTER the terminal event must not report a served response as failed.

    `_consume` keeps iterating past `final` -- the generator still has to finish and close -- so an
    `aclose` that raises arrives with the completion already delivered. Charging it to the request
    turns a served response into an engine error, inflating the error rate under exactly the slow
    cleanup the `final_at` timestamp exists to keep out of latency.
    """
    import time

    from flash.serving.bench import driver
    from flash.serving.bench.metrics import ERROR_ENGINE

    origin = time.monotonic()

    class _FailsAfterFinal:
        def _stream_generate(self, payload, forwarded, _lora, _gid):  # type: ignore[no-untyped-def]
            async def _gen():
                yield {"type": "ready", "engine_replica_id": "r0"}
                yield {"type": "delta", "text": "hi"}
                yield {"type": "choice_finished", "finish_reason": "stop"}
                yield {
                    "type": "final",
                    "prompt_tokens": 512,
                    "completion_tokens": 8,
                    "cached_tokens": 0,
                    "cached_tokens_reported": True,
                }
                raise RuntimeError("aclose exploded")

            return _gen()

    record = asyncio.run(
        driver.run_request(
            engine=_FailsAfterFinal(),
            base_model="acme/model",
            bucket="b",
            concurrency=1,
            block=0,
            uid="u2",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            origin=origin,
            expected_prompt_tokens=512,
            bucket_target_tokens=512,
        )
    )

    assert record.error is None, (
        "a completion delivered before a failing cleanup is reported as a failed request "
        f"({record.error}: {record.error_detail})"
    )
    assert record.ok, "the delivered completion was never validated"
    # The unhealthy stream is not absorbed into the success -- it is published on its own channel.
    assert record.cleanup_error == ERROR_ENGINE
    assert "aclose exploded" in (record.cleanup_error_detail or "")


def test_a_cleanup_fault_is_published_without_entering_the_error_rate() -> None:
    """R24: a served-then-unhealthy request counts as a success AND stays visible.

    Folding the fault into `error_rate` would put the harness back to reporting delivered
    completions as failures; dropping it entirely would hide an unhealthy container behind a clean
    cell. The reduction has to do both.
    """
    from flash.serving.bench.metrics import ERROR_ENGINE, RequestRecord, reduce_cell

    def _served(uid: str, cleanup: str | None) -> RequestRecord:
        record = RequestRecord(
            uid=uid,
            base_model="acme/model",
            bucket="b",
            concurrency=1,
            block=0,
            started_at=0.0,
            first_token_at=0.1,
            finished_at=1.0,
            prompt_tokens=512,
            completion_tokens=8,
            cached_tokens=0,
            cached_tokens_reported=True,
            finish_reason="stop",
            ok=True,
        )
        record.cleanup_error = cleanup
        return record

    cell = reduce_cell(
        [_served("a", None), _served("b", ERROR_ENGINE), _served("c", ERROR_ENGINE)],
        base_model="acme/model",
        bucket="b",
        concurrency=1,
        block=0,
        wall_seconds=10.0,
    )

    assert cell.succeeded == 3, "a cleanup fault removed a delivered completion from the successes"
    assert cell.error_rate == 0.0, "a cleanup fault entered the error rate"
    assert cell.cleanup_faults == 2, (
        "cleanup faults are not published, so an unhealthy cell reads clean"
    )
    assert cell.cleanup_breakdown == {ERROR_ENGINE: 2}


def test_the_execution_digest_covers_the_drain_reap_interval() -> None:
    """R24: `_drain` is digested, but its source only NAMES the reap interval.

    Retuning it changes how long a cancelled request may clean up and whether the container is
    terminated -- so which timeout records and which bucket artifacts survive a drain -- while every
    digested source stays byte-identical.
    """
    from flash.serving.bench import workload

    assert "_DRAIN_REAP_SECONDS" in workload._DRIVER_CONSTANTS, (
        "the reap interval is not digested, so two campaigns with different drain cleanup "
        "behaviour publish the same workload_checksum"
    )


def test_every_artifact_records_the_applied_vllm_patch() -> None:
    """R24: the image REWRITES installed vLLM files, and no version string reports it.

    `runtime_packages` reads distribution metadata, which the patch does not touch, so two images
    built from different patch contents resolve identical versions, checkpoints and workload
    checksums while measuring different kernel and LoRA execution.
    """
    import hashlib

    tree = ast.parse(BENCH_APP.read_text(encoding="utf-8"))
    writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_write_artifact"
    ]
    assert writes, "the bench script writes no artifacts"
    for call in writes:
        payload = call.args[0]
        assert isinstance(payload, ast.Dict), (
            f"artifact payload at line {call.lineno} is not a dict"
        )
        keys = {k.value for k in payload.keys if isinstance(k, ast.Constant)}
        # `None` keys are `**` spreads. The frozen `local_identity` mapping carries
        # `serving_patch`, so a payload spreading it satisfies this contract.
        spreads = any(k is None for k in payload.keys)
        assert "serving_patch" in keys or spreads, (
            f"the artifact written at line {call.lineno} cannot say which vLLM patch it measured"
        )

    # `REPO_DIR` is derived from the script's `__file__`, which an exec'd node does not have, so it
    # is injected as the real checkout root rather than lifted.
    namespace = _bench_namespace(
        "_serving_patch_identity", "MOE_LORA_PATCH", REPO_DIR=REPO_ROOT, hashlib=hashlib, Path=Path
    )
    identity = namespace["_serving_patch_identity"]()
    patch = REPO_ROOT / "docker" / "patch_vllm_moe_lora.py"
    assert identity["path"] == "docker/patch_vllm_moe_lora.py"
    assert identity["sha256"] == hashlib.sha256(patch.read_bytes()).hexdigest(), (
        "the recorded identity does not digest the patch the image actually applies"
    )


def test_the_execution_digest_covers_the_forwarded_base_model_record() -> None:
    """R22: the record that routes generation must move the checksum.

    `run_request` is digested but only NAMES `base_model_record`. Flipping `serve_base_model` sends
    every request down the adapter path, and flipping `thinking` changes the default the engine
    applies to every completion -- materially different generation work under a checksum that never
    moved.
    """
    from flash.serving.bench import driver
    from flash.serving.bench import workload as workload_module

    assert "base_model_record" in workload_module._DRIVER_SOURCES, (
        "the forwarded registration is not part of the driver source digest"
    )

    before = workload_checksum()
    original = driver.base_model_record

    def _rerouted(base_model: str) -> dict[str, object]:
        record = original(base_model)
        record["serve_base_model"] = False
        return record

    try:
        driver.base_model_record = _rerouted  # type: ignore[assignment]
        assert workload_checksum() != before, (
            "rerouting every request through the adapter path left the checksum unchanged"
        )
    finally:
        driver.base_model_record = original  # type: ignore[assignment]


def test_the_probe_records_the_runtime_versions_that_can_move_the_curve() -> None:
    """R22: Torch and CUDA alone do not identify this runtime.

    The image installs from `pyproject.toml` ranges rather than a lockfile, so two builds of the
    same commit can resolve different Transformers, tokenizers or kernel libraries -- changing
    prompt assembly or execution -- while every checksum, checkpoint and catalog row matches.
    """
    from flash.serving.bench import probe as probe_module

    for name in ("transformers", "tokenizers", "vllm", "huggingface-hub"):
        assert name in probe_module._RUNTIME_PACKAGES, (
            f"{name} can change the measured curve but is not recorded"
        )

    resolved = probe_module.probe_runtime_packages()
    assert set(resolved) == set(probe_module._RUNTIME_PACKAGES), (
        "the probe does not report every package it declares"
    )
    # Fail-soft: an absent distribution is recorded as None, never invented and never raised.
    assert resolved["transformers"] is None or isinstance(resolved["transformers"], str)

    source = inspect.getsource(probe_module.probe_all)
    assert '"runtime_packages"' in source, "probe_all omits the resolved runtime identity"


def test_the_summary_carries_the_checkpoint_the_sweep_accepted() -> None:
    """R22: the combined envelope must name the weights that produced its curves.

    `engine_catalog.immutable_revisions` is empty for the two unpinned models, so without the
    accepted commits the summary -- the artifact that presents the envelope -- identifies its
    weights only by a mutable repository name.
    """
    source = BENCH_APP.read_text()
    tree = ast.parse(source)
    gate = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_require_one_identity"
    )
    returns = [
        node
        for node in ast.walk(gate)
        if isinstance(node, ast.Return)
        and node.value is not None
        and not _is_empty_dict(node.value)
    ]
    assert returns, "the gate discards the identity it accepted instead of returning it"

    sweep_body = source[source.index("def _run_sweep_lane(") :]
    assert "accepted_checkpoint = _require_one_identity(" in sweep_body, (
        "the accepted identity is not captured from the gate"
    )
    summary = sweep_body[sweep_body.index('f"summary-') - 2000 : sweep_body.index('f"summary-')]
    assert '"accepted_checkpoint": accepted_checkpoint' in summary, (
        "the summary payload omits the checkpoint the sweep accepted"
    )
    assert '"runtime_packages"' in summary, "the summary payload omits the resolved runtime"


def _is_empty_dict(node: ast.expr) -> bool:
    return isinstance(node, ast.Dict) and not node.keys


def test_the_tokenizer_commit_names_the_tree_the_prompts_were_built_from() -> None:
    """R23: the ref must not stand in for the tokenizer this process actually loaded.

    The parent loads its tokenizer in `load_engine_config`, BEFORE `AsyncLLMEngine.from_engine_args`
    downloads the weights. On the 35B, model and tokenizer share one unpinned repository, so the
    engine's own fetch can land a second snapshot and advance `refs/main` to it. Both snapshots then
    hold tokenizer files and the ref names the newer one -- and because the ref is among the
    holders, the resolver used to return it under a source the publication gate ACCEPTS. The
    measured tokens were produced by the older tree, so that is false provenance rather than
    ambiguity, and nothing downstream could tell the difference.
    """
    from flash.serving.bench import probe as probe_module

    loaded_commit = "1111111111111111111111111111111111111111"
    advanced_commit = "2222222222222222222222222222222222222222"

    # The vocabulary the PARENT's tokenizer holds. Only `model.vocab` is compared, because
    # `tokenizers` re-normalizes the neighbouring fields on load.
    loaded_vocab = {"hello": 0, "world": 1}
    advanced_vocab = {"hello": 0, "world": 1, "later": 2}

    class _Backend:
        def to_str(self) -> str:
            return json.dumps({"model": {"vocab": loaded_vocab}})

    class _LoadedTokenizer:
        backend_tokenizer = _Backend()

    with tempfile.TemporaryDirectory() as cache:
        repos = {
            role: entry["repo"]
            for role, entry in probe_module.probe_resolved_revisions("Qwen/Qwen3.6-35B-A3B").items()
        }
        assert repos["model"] == repos["tokenizer"], (
            "the 35B no longer shares one repository across roles, so this race needs rechecking"
        )
        root = Path(cache) / f"models--{repos['tokenizer'].replace('/', '--')}"

        for commit, vocab in ((loaded_commit, loaded_vocab), (advanced_commit, advanced_vocab)):
            snapshot = root / "snapshots" / commit
            snapshot.mkdir(parents=True)
            (snapshot / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab}}))
        refs = root / "refs"
        refs.mkdir(parents=True)
        # The engine's later download advanced the ref past the tree the parent loaded.
        (refs / "main").write_text(f"{advanced_commit}\n")

        with mock.patch("huggingface_hub.constants.HF_HUB_CACHE", cache):
            resolved = probe_module.probe_resolved_revisions(
                "Qwen/Qwen3.6-35B-A3B", _LoadedTokenizer()
            )
            # The load-bearing arm: with no tokenizer to compare against, the ref is all the cache
            # can offer and the answer must NOT be a confident one.
            blind = probe_module.probe_resolved_revisions("Qwen/Qwen3.6-35B-A3B")

    assert resolved["tokenizer"]["commit"] == loaded_commit, (
        "the tokenizer commit names the tree the ENGINE's download advanced to, so the published "
        "curve credits its prompts to a tokenizer they were never built with"
    )
    assert resolved["tokenizer"]["source"] == "snapshot-contents-verified", (
        "a content-matched commit must be reported under its own source, so an artifact can say "
        "the identification was verified rather than inferred from a repository-wide ref"
    )
    assert blind["tokenizer"]["commit"] != loaded_commit, (
        "the ref alone identified the loaded tree, so this cache no longer reproduces the race and "
        "the content match above proves nothing"
    )
    assert blind["tokenizer"]["source"] != "snapshot-contents-verified", (
        "a probe with no loaded tokenizer claimed a verified match, so the source label does not "
        "depend on the comparison it names"
    )


def test_probe_all_forwards_the_tokenizer_the_driver_fits_prompts_with() -> None:
    """R23: the content match is only reachable if the engine's own tokenizer is passed to it.

    `probe_all` is what every lane calls. A resolver that can disambiguate but is never handed the
    loaded object would leave the defect exactly where it was, with a helper that looks like a fix.
    """
    from flash.serving.bench import probe as probe_module

    seen: list[Any] = []

    def _record(base_model: str, tokenizer: Any = None) -> dict[str, Any]:
        seen.append(tokenizer)
        return {}

    sentinel = object()

    class _Engine:
        base_model = "Qwen/Qwen3.6-35B-A3B"
        tokenizer = sentinel

    with (
        mock.patch.object(probe_module, "probe_resolved_revisions", _record),
        mock.patch.object(probe_module, "probe_runtime_packages", return_value={}),
        mock.patch.object(probe_module, "probe_gpu", return_value={}),
        mock.patch.object(probe_module, "probe_gdn_backend", return_value={}),
        mock.patch.object(probe_module, "probe_engine_kv_cache", return_value={}),
    ):
        probe_module.probe_all("Qwen/Qwen3.6-35B-A3B", _Engine())

    assert seen == [sentinel], (
        "probe_all did not forward the engine's tokenizer, so the loaded tree can never be "
        "content-matched and the tokenizer commit falls back to the repository-wide ref"
    )


def test_the_reservation_covers_the_method_timeout_headroom_it_authorizes() -> None:
    """R23: Modal enforces TIMEOUT_SECONDS, which exceeds every phase the estimators price.

    `TIMEOUT_SECONDS` is the worst-case bucket PLUS `TIMEOUT_HEADROOM_SECONDS`. That slack is
    granted per call and billable before Modal terminates it, so a reservation stopping at the named
    phases lets a sweep bill the headroom once per separately bootable call beyond its ceiling --
    the one direction a budget must never err.

    Measured by re-pricing the SAME estimator with the headroom constant zeroed, so the guard reads
    the term out of the real arithmetic instead of restating it. A reservation that ignores the
    constant produces an identical number both ways and fails here.
    """

    def _namespace(headroom: float | None) -> dict[str, Any]:
        injected: dict[str, Any] = {
            "REQUEST_TIMEOUT_SECONDS": REQUEST_TIMEOUT_SECONDS,
            "bench_engine_overrides_for": bench_engine_overrides_for,
            "concurrency_grid": concurrency_grid,
            "prompt_fit_seconds_bound": prompt_fit_seconds_bound,
            # Held FIXED across both namespaces: the headroom delta must isolate the headroom, so
            # every other term has to price identically on each side.
            "WARMUP_FIT_SECONDS_BOUND": _WARMUP_FIT_BOUND,
            "_FUNDED_WARMUP_FIT_SECONDS": _WARMUP_FIT_BOUND + fitting_watchdog_grace_seconds(),
        }
        names = [
            "_canary_gpu_seconds_estimate",
            "_sweep_gpu_seconds_estimate",
            "SCALEDOWN_WINDOW_SECONDS",
            "STARTUP_TIMEOUT_SECONDS",
            "PROBE_TIMEOUT_SECONDS",
        ]
        if headroom is None:
            names.append("TIMEOUT_HEADROOM_SECONDS")
        else:
            injected["TIMEOUT_HEADROOM_SECONDS"] = headroom
        return _bench_namespace(*names, **injected)

    real = _namespace(None)
    headroom = float(real["TIMEOUT_HEADROOM_SECONDS"])
    assert headroom > 0, "the headroom is zero, so this guard can no longer detect its omission"
    zeroed = _namespace(0.0)

    model = "Qwen/Qwen3.5-9B"
    for selected in ([BUCKETS_BY_NAME["near_32k"]], list(BUCKETS)):
        charged = real["_sweep_gpu_seconds_estimate"](model, selected) - zeroed[
            "_sweep_gpu_seconds_estimate"
        ](model, selected)
        # Once per separately bootable call: every bucket's `run_bucket.remote()` plus the canary's
        # `certify.remote()`, each of which carries its own method timeout.
        assert charged == pytest.approx(headroom * (len(selected) + 1)), (
            f"the sweep over {len(selected)} bucket(s) reserves {charged}s of method-timeout "
            f"headroom but its calls are permitted to bill {headroom * (len(selected) + 1)}s, so "
            "an accepted sweep can spend past its ceiling"
        )

    canary_charged = (
        real["_canary_gpu_seconds_estimate"]() - zeroed["_canary_gpu_seconds_estimate"]()
    )
    assert canary_charged == pytest.approx(headroom), (
        "the canary lane reserves less than the method-timeout headroom its single call is "
        "permitted to bill, so an accepted canary can exceed its own ceiling"
    )


# ── Round-25: cancelled cleanup after a delivered final, and the warmup contract ───────────────


def test_a_cancellation_after_the_final_keeps_the_served_completion() -> None:
    """R25: a completion already delivered must not become an error because the drain cancelled it.

    `_drain` cancels whatever is still pending at the cell bound, and a request whose `final` event
    already arrived can be sitting in generator cleanup at that instant -- the response was served,
    the stream simply had not finished closing. `CancelledError` derives from `BaseException`, so
    the `except Exception` arm in `run_request` never saw it: the cancellation escaped the function
    entirely, `_drain` got no record, and the synthetic timeout it substitutes counted an
    already-served completion as a failure. That inflates the error rate under exactly the slow
    cleanup the delivered-final routing exists to keep out of the numbers, and it inflates it in the
    direction a capacity claim must never err.

    Both halves are asserted because they are separate defects: `run_request` must swallow the
    cancellation, AND `_drain` must prefer the task's own record over its synthetic one. Fixing
    either alone still discards the completion.
    """
    from flash.serving.bench import driver as driver_module

    origin = time.monotonic()

    class _StallsAfterFinal:
        """Delivers a complete response, then hangs in generator cleanup.

        This is the real shape of the defect: the `final` event has ARRIVED, so the completion was
        served, and the drain's cancel then lands while the generator is still closing.
        """

        def _stream_generate(self, payload, forwarded, _lora, _gid):  # type: ignore[no-untyped-def]
            async def _gen():
                yield {"type": "ready"}
                yield {"type": "token", "text": "hi"}
                yield {"type": "choice_finished", "finish_reason": "stop"}
                yield {
                    "type": "final",
                    "prompt_tokens": 512,
                    "completion_tokens": 8,
                    "cached_tokens": 0,
                    "cached_tokens_reported": True,
                }
                # The stall the drain cancels into.
                await asyncio.sleep(3600)

            return _gen()

    async def _exercise() -> list[RequestRecord]:
        # The REAL `run_request`, not a stand-in: the swallow under test lives inside it, and a fake
        # coroutine would prove only the drain half of the fix.
        task = asyncio.ensure_future(
            driver_module.run_request(
                engine=_StallsAfterFinal(),
                base_model="Qwen/Qwen3.5-9B",
                bucket="short_interactive",
                concurrency=1,
                block=0,
                uid="uid-final",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                origin=origin,
                expected_prompt_tokens=512,
                bucket_target_tokens=512,
            )
        )
        # Let the stream run to its `final` and settle into the stall. Cancelling before that would
        # be the OTHER case -- nothing delivered -- which the next test covers.
        await asyncio.sleep(0.05)
        # Presented to the drain as still pending, which is what puts it on the cancel-and-reap
        # path where the synthetic timeout used to be substituted unconditionally.
        with mock.patch.object(
            driver_module.asyncio, "wait", mock.AsyncMock(return_value=(set(), {task}))
        ):
            return await driver_module._drain(
                {task},
                base_model="Qwen/Qwen3.5-9B",
                bucket="short_interactive",
                concurrency=1,
                block=0,
                spawned_at={},
                spawned_uid={},
            )

    drained = asyncio.run(_exercise())

    assert len(drained) == 1, "the drain lost the record the task carried"
    kept = drained[0]
    assert kept.ok is True, (
        "a completion that was served and then cancelled during stream cleanup was recorded as a "
        "failure; the error rate now counts a delivered response against the engine"
    )
    assert kept.error is None, f"the served completion carries error {kept.error!r}"
    assert kept.uid == "uid-final", "the synthetic record displaced the task's own evidence"
    # The unhealthy close is still PUBLISHED rather than absorbed into a clean success.
    assert kept.cleanup_error == ERROR_ENGINE, "the cleanup fault was swallowed with the record"


def test_a_cancellation_with_nothing_delivered_still_counts_as_a_failure() -> None:
    """R25: the converse. Swallowing every cancellation would hide genuine mid-stream kills.

    A request cancelled before its `final` arrived is a real cancellation: nothing was served, and
    the record must stay a failure. `run_request` re-raises in that case, so `result()` raises in
    the drain and the synthetic record is what the attempt gets -- which keeps the request in the
    denominator instead of deleting it. Without this direction the fix above would read as green
    while quietly converting every cancelled request into a success.
    """
    from flash.serving.bench import driver as driver_module

    origin = time.monotonic()

    class _StallsBeforeFinal:
        """Opens the stream and then hangs, with no terminal event ever delivered."""

        def _stream_generate(self, payload, forwarded, _lora, _gid):  # type: ignore[no-untyped-def]
            async def _gen():
                yield {"type": "ready"}
                yield {"type": "token", "text": "hi"}
                await asyncio.sleep(3600)

            return _gen()

    async def _exercise() -> list[RequestRecord]:
        # Again the REAL `run_request`, so the re-raise under test is the one that actually runs.
        task = asyncio.ensure_future(
            driver_module.run_request(
                engine=_StallsBeforeFinal(),
                base_model="Qwen/Qwen3.5-9B",
                bucket="short_interactive",
                concurrency=1,
                block=0,
                uid="uid-nothing",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                origin=origin,
                expected_prompt_tokens=512,
                bucket_target_tokens=512,
            )
        )
        await asyncio.sleep(0.05)
        with mock.patch.object(
            driver_module.asyncio, "wait", mock.AsyncMock(return_value=(set(), {task}))
        ):
            return await driver_module._drain(
                {task},
                base_model="Qwen/Qwen3.5-9B",
                bucket="short_interactive",
                concurrency=1,
                block=0,
                spawned_at={"__missing__": 0.0},  # type: ignore[dict-item]
                spawned_uid={},
            )

    drained = asyncio.run(_exercise())

    assert len(drained) == 1, "a cancelled request was DELETED from the attempt denominator"
    assert drained[0].ok is False, (
        "a request cancelled with nothing delivered was recorded as a success, so a cell that "
        "ended mid-stream reports a cleaner error rate than it earned"
    )
    assert drained[0].error == ERROR_TIMEOUT
    assert drained[0].uid != "uid-nothing", (
        "the request's own record survived a cancellation that delivered nothing, so the "
        "swallow above is unconditional and every cancelled request now reads as served"
    )


def test_the_warmup_contract_is_inside_the_execution_digest() -> None:
    """R25: what CONDITIONS every measured cell must move the checksum.

    The warmup is what moves compilation and lazy-initialization cost out of the measured window,
    so its request count, prompt shape and sequencing decide which startup cost the published curve
    excludes. It ran before every canary and every cold replacement while living in the entrypoint
    script -- which `_execution_digest` does not read at all -- so going from five warmups to one,
    or warming concurrently instead of sequentially, left `workload_checksum` byte-identical. Two
    campaigns that excluded materially different startup costs then compared as if they had
    measured the same thing.

    Digested through its OWN tuples rather than the driver's, because the warmup imports from the
    driver: re-exporting it back through the driver to reuse `_DRIVER_SOURCES` would be a cycle.
    The count is digested BY VALUE rather than by source, because `run_warmup`'s own source only
    names the number its caller passes.
    """
    from flash.serving.bench import warmup as warmup_module
    from flash.serving.bench import workload

    assert "run_warmup" in workload._WARMUP_SOURCES, (
        "the warmup that precedes every measured cell is outside the execution digest, so its "
        "prompt shape and sequencing can change without moving the checksum"
    )
    assert "CANARY_WARMUP_REQUESTS" in workload._WARMUP_CONSTANTS, (
        "the warmup COUNT is not digested; `run_warmup` names it rather than revealing it, so "
        "five warmups and one warmup produce the same checksum"
    )

    # Not merely listed: the digest must actually MOVE when either one changes.
    baseline = workload.workload_checksum()
    with mock.patch.object(warmup_module, "CANARY_WARMUP_REQUESTS", 1):
        assert workload.workload_checksum() != baseline, (
            "the warmup count is named in _WARMUP_CONSTANTS but does not move the checksum"
        )

    async def _different_warmup(engine: Any, requests: int) -> dict[str, Any]:
        return {"warmups": [], "assembled_prompt_tokens": 0}

    with mock.patch.object(warmup_module, "run_warmup", _different_warmup):
        assert workload.workload_checksum() != baseline, (
            "the warmup body is named in _WARMUP_SOURCES but does not move the checksum"
        )


def test_every_call_is_bounded_at_what_its_own_phases_reserve() -> None:
    """No remote call may be authorized to run longer than the reservation that funded it.

    `TIMEOUT_SECONDS` is one class-wide number derived from the WIDEST bucket, and Modal applies it
    to every method. So a `short_interactive` call was permitted to bill 19439s while the estimator
    priced it at 13989s, and the canary 19439s against 5309s -- thousands of authorized-but-
    unreserved GPU-seconds per call, in the one direction a budget must never err.

    The repair bounds each call at its OWN priced phases rather than raising the reservation to
    cover slack the call's bounds make unreachable (that would have inflated a canary from $4.89 to
    $12.06 and refused runs that cannot cost that much). This asserts the property that makes the
    ceiling real: for every bucket and every grid width, the bound the call is held to is exactly
    what the estimator reserves for it, and never the class-wide grant.
    """
    namespace = _bench_namespace(
        "bucket_call_priced_seconds",
        "certify_call_priced_seconds",
        "_worst_case_bucket_seconds",
        "TIMEOUT_SECONDS",
        "TIMEOUT_HEADROOM_SECONDS",
        "PROBE_TIMEOUT_SECONDS",
        "WARMUP_FIT_SECONDS_BOUND",
        "_FUNDED_WARMUP_FIT_SECONDS",
        bench_engine_overrides_for=bench_engine_overrides_for,
        concurrency_grid=concurrency_grid,
        prompt_fit_seconds_bound=prompt_fit_seconds_bound,
        REQUEST_TIMEOUT_SECONDS=REQUEST_TIMEOUT_SECONDS,
        BENCH_MODELS=BENCH_MODELS,
        BUCKETS=BUCKETS,
    )
    priced_bucket = namespace["bucket_call_priced_seconds"]
    priced_certify = namespace["certify_call_priced_seconds"]
    class_timeout = namespace["TIMEOUT_SECONDS"]

    points = len(list(concurrency_grid(8)))
    # The canary is the widest gap: its phases are a probe plus five warmups, against a ceiling
    # derived from a full near_32k grid.
    assert priced_certify() < class_timeout, (
        "the canary's bound must be its own phases, not the class-wide grant"
    )
    for bucket in BUCKETS:
        bound = priced_bucket(bucket, points)
        assert bound <= class_timeout, (
            f"{bucket.name} is priced above the class timeout that terminates it"
        )
    # The narrow buckets must be bounded STRICTLY below the widest, or the bound is the class-wide
    # grant wearing a different name and the defect is intact.
    widest = max(priced_bucket(bucket, points) for bucket in BUCKETS)
    narrow = min(priced_bucket(bucket, points) for bucket in BUCKETS)
    assert narrow < widest, (
        "every bucket priced identically means the bound is not per-call; a short bucket would "
        "again be authorized to spend the widest bucket's ceiling"
    )
    # And the ceiling itself is still derived from the same terms, so widening a bucket moves the
    # bound and the reservation together instead of reopening the gap.
    assert namespace["_worst_case_bucket_seconds"]() == pytest.approx(widest)


def test_the_bounded_call_ends_the_container_rather_than_billing_past_its_bound() -> None:
    """A call that outlives its priced bound must stop billing, not return and leave work running.

    Raising would let the coroutine stay pinned in an uninterruptible engine call, still on the GPU
    through the scaledown window, still spending against a reservation it has already exceeded. The
    file ends the process for exactly this in two other places; this asserts the new bound does too,
    driving the real helper rather than a stand-in.
    """
    namespace = _bench_namespace(
        "_within_call_bound",
        asyncio=asyncio,
        contextlib=contextlib,
        os=os,
        sys=sys,
        drain_reap_seconds=lambda: 0.01,
    )
    within = namespace["_within_call_bound"]

    exits: list[int] = []
    namespace["os"] = types.SimpleNamespace(_exit=lambda code: exits.append(code))

    # Uncooperative on purpose: this stands in for a coroutine pinned in an uninterruptible engine
    # call, which is the case the exit exists for. A task that STOPS when cancelled has stopped
    # spending, and the bound correctly declines to kill the container for it -- that case is
    # covered by `test_the_bounded_call_reaps_work_when_its_own_frame_is_cancelled`.
    async def _stalls() -> str:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(30)
        return "never"

    async def _returns() -> str:
        return "done"

    # The fast path is untouched: a call inside its bound returns its own value.
    assert asyncio.run(within(_returns(), 5.0, "fast")) == "done"

    # The slow path ends the container instead of returning past the bound. The real `os._exit`
    # halts the process here; the fake above only records, so execution continues into the
    # trailing `raise` that exists for the case where the work DID stop during the reap. Catching
    # it keeps the assertion below about the exit, which is the property under test.
    with contextlib.suppress(TimeoutError):
        asyncio.run(within(_stalls(), 0.05, "stalled"))
    assert exits == [75], (
        "a call that outlived its priced bound must end the container; returning or raising leaves "
        "it billing on the GPU against a reservation it already exceeded"
    )


def test_the_bounded_call_reaps_work_when_its_own_frame_is_cancelled() -> None:
    """The bound must close on the cancellation door too, not only the timeout door.

    `asyncio.shield` is what lets the bound reap: it keeps the inner task alive so the wrapper can
    cancel it deliberately. But shield cuts both ways -- when the OUTER frame is cancelled (a lane
    torn down, a Modal timeout unwinding the caller) the `wait_for` raises `CancelledError`, not
    `TimeoutError`, and the shielded task survives the frame that was supposed to bound it. Catching
    only `TimeoutError` therefore reaped nothing on that route: the frame unwound, the task stayed
    pinned in the engine, and the container kept billing against a reservation nobody was watching.

    Also asserts the two conditions that keep the reap honest: a task that FINISHES during the reap
    must not kill the container -- it stopped spending, and killing then destroys the artifact of
    work that completed inside its reservation -- and the cancellation must re-raise so the caller
    still observes it.
    """
    namespace = _bench_namespace(
        "_within_call_bound",
        asyncio=asyncio,
        contextlib=contextlib,
        os=os,
        sys=sys,
        drain_reap_seconds=lambda: 0.05,
    )
    within = namespace["_within_call_bound"]

    exits: list[int] = []
    namespace["os"] = types.SimpleNamespace(_exit=lambda code: exits.append(code))

    # A task that ignores cancellation, exactly like a coroutine pinned in an uninterruptible
    # engine call: cancelling it does not stop it spending.
    async def _ignores_cancellation() -> str:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(30)
        return "never"

    async def _cancel_the_outer_frame() -> None:
        outer = asyncio.ensure_future(within(_ignores_cancellation(), 60.0, "cancelled"))
        await asyncio.sleep(0.05)
        outer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await outer

    asyncio.run(_cancel_the_outer_frame())
    assert exits == [75], (
        "a bounded call whose own frame is cancelled must still reap the work it started; "
        "`shield` keeps that task alive, so leaving it running keeps billing the GPU after the "
        "frame that authorized it is gone"
    )

    # A task that stops when asked has stopped spending. Killing the container then would destroy
    # the artifact of work that completed inside its reservation.
    exits.clear()
    raised: list[str] = []

    async def _stops_when_asked() -> str:
        await asyncio.sleep(30)
        return "never"

    async def _cancel_a_cooperative_task() -> None:
        outer = asyncio.ensure_future(within(_stops_when_asked(), 60.0, "cooperative"))
        await asyncio.sleep(0.05)
        outer.cancel()
        try:
            await outer
        except asyncio.CancelledError:
            raised.append("cancelled")

    asyncio.run(_cancel_a_cooperative_task())
    assert exits == [], (
        "a task that stopped when cancelled is no longer spending; ending the container then "
        "destroys the evidence of work that finished inside its reservation"
    )
    assert raised == ["cancelled"], (
        "the bound accounts for the cancellation, it does not swallow it; the caller must still "
        "observe that its frame was cancelled"
    )


def test_a_configuration_only_snapshot_cannot_answer_which_weights_ran() -> None:
    """`config.json` alone must not qualify a snapshot as holding the model.

    `_snapshots_holding` matches its role markers with `any(...)`, and `config.json` was a model
    marker -- but it is the one file every metadata-only fetch pulls, including this probe's own
    `AutoConfig.from_pretrained`. So a snapshot containing configuration and no weights entered the
    model candidates. On the two UNPINNED repositories, a repo that advances after the weights
    download and a later config or tokenizer fetch is enough to create one; if `refs/main` then
    names it, it is accepted as the commit the running weights came from.

    A capacity number is attributed to weights, so the model role must require a file that actually
    carries them.
    """
    from flash.serving.bench.probe import _ROLE_MARKERS, _snapshots_holding

    assert "config.json" not in _ROLE_MARKERS["model"], (
        "`config.json` is present in every metadata-only snapshot, so accepting it as a model "
        "marker lets a weightless directory answer which weights ran"
    )

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        # The real shape: an older snapshot carrying weights, a newer one carrying only the
        # configuration and tokenizer a later fetch pulled.
        weights = root / "snapshots" / "aaaaaaaa"
        weights.mkdir(parents=True)
        (weights / "config.json").write_text("{}", encoding="utf-8")
        (weights / "model.safetensors").write_text("weights", encoding="utf-8")

        metadata_only = root / "snapshots" / "bbbbbbbb"
        metadata_only.mkdir(parents=True)
        (metadata_only / "config.json").write_text("{}", encoding="utf-8")
        (metadata_only / "tokenizer.json").write_text("{}", encoding="utf-8")

        holding = _snapshots_holding(root, "model")
        assert holding == ["aaaaaaaa"], (
            f"only the snapshot carrying weights may answer for the model role; got {holding!r}, "
            "which lets a configuration-only tree be published as the measured checkpoint"
        )
        # The tokenizer role is unchanged: it reads the tree that actually holds tokenizer files.
        assert _snapshots_holding(root, "tokenizer") == ["bbbbbbbb"]


def test_the_stream_accumulators_initial_state_is_digested() -> None:
    """The defaults that decide validation must move the checksum.

    `_absorb_event` and `_validate` are digested, but half of what they decide lives in the
    accumulator's INITIAL state. `_validate` rejects on `cached_tokens_reported is not True` and on
    `saw_final` being false, so seeding either differently -- or pre-populating a usage field --
    changes which requests become errors, and the error rate is the numerator this campaign
    publishes. With `_StreamOutcome` outside `_DRIVER_SOURCES`, two campaigns applying materially
    different validation compared as compatible.
    """
    from flash.serving.bench import workload

    assert "_StreamOutcome" in workload._DRIVER_SOURCES, (
        "the accumulator's initial state decides which requests validate, so it must be digested "
        "alongside the functions that read it"
    )

    # Not merely listed: the digest must actually change when a default changes.
    driver_source = (REPO_ROOT / "flash" / "serving" / "bench" / "driver.py").read_text(
        encoding="utf-8"
    )
    assert "self.saw_final = False" in driver_source

    real_getsource = inspect.getsource

    def _drift(obj: Any) -> str:
        text = real_getsource(obj)
        if getattr(obj, "__name__", "") == "_StreamOutcome":
            text = text.replace("self.saw_final = False", "self.saw_final = True")
        return text

    before = workload.workload_checksum()
    with mock.patch.object(inspect, "getsource", _drift):
        after = workload.workload_checksum()
    assert before != after, (
        "changing the accumulator must move `workload_checksum()`; if it does not, a campaign with "
        "different initial validation state can claim compatibility with this one"
    )


def test_the_local_identities_are_frozen_before_the_first_remote_call() -> None:
    """Provenance must describe the sources the image uploaded, not the checkout hours later.

    Modal uploads the local sources when the engine object is constructed; a sweep then runs for
    hours. Recomputing `workload_checksum()` and the source digests when `run_bucket.remote()`
    RETURNS records whatever the working tree says at that moment -- an edit, a branch switch, a
    `git pull` during the paid run -- and attributes it to code the container is no longer running.
    That is worse than no provenance: it is confidently wrong.
    """
    source = BENCH_APP.read_text(encoding="utf-8")

    frozen_at = source.index("local_identity = {")
    first_remote = source.index("_run_canary(base_model, engine, expected_gpu)")
    assert frozen_at < first_remote, (
        "the local identities must be captured before the first remote call, or they describe a "
        "checkout that may have moved while the paid sweep ran"
    )

    # Every artifact spreads the frozen mapping; none recomputes. The envelope in `_Lane.write`
    # applies it to all four artifacts (canary, per-bucket, summary, failure), so this asserts the
    # single spread AND that no write bypasses it -- a stronger contract than four repeated
    # spreads, where dropping one was invisible until that artifact was the only evidence left.
    assert source.count("**self.local_identity") == 1, (
        "the artifact envelope no longer spreads the frozen identities"
    )
    direct = [
        line
        for line in source.splitlines()
        if "_write_artifact(" in line and "def _write_artifact" not in line
    ]
    assert len(direct) == 1, (
        f"every artifact must be written through the envelope; found {len(direct)} direct calls, "
        "one of which would record post-run local state"
    )
    body = source[source.index("local_identity = {") :]
    assert body.count("workload_checksum()") == 1, (
        "`workload_checksum()` must be evaluated once, into the frozen mapping; a second call site "
        "inside the lane re-reads the checkout after remote work has already run"
    )


def test_the_benchmark_scheduler_has_a_recorded_content_identity() -> None:
    """The loop that decides which cells run must be covered by some recorded identity.

    `workload_checksum()` digests named objects from `flash.serving.bench` and
    `_serving_source_identity()` digests the `flash` package. Neither can see `scripts/` -- yet
    `_run_bucket` lives there and owns the concurrency ordering, the warm engine each cell
    inherits, and the bucket, block and invocation forwarded into `run_cell`. Changing any of them
    changes the prompts issued and the cells the curve contains while every other recorded identity
    stays byte-identical.
    """
    namespace = _bench_namespace(
        "_harness_source_identity",
        Path=Path,
        hashlib=hashlib,
        __file__=str(BENCH_APP),
    )
    identity = namespace["_harness_source_identity"]()

    assert identity.get("sha256") == hashlib.sha256(BENCH_APP.read_bytes()).hexdigest(), (
        "the harness identity must digest the script's real content; a stale or hand-maintained "
        "value moves only when someone remembers to move it"
    )

    # And it must actually be recorded, not merely computable.
    source = BENCH_APP.read_text(encoding="utf-8")
    assert '"harness_source": _harness_source_identity()' in source, (
        "the scheduler's identity must reach the artifacts, or the curve cannot say which loop "
        "produced its cells"
    )


def test_one_envelope_cannot_span_two_kernel_dispatch_stacks() -> None:
    """Buckets that ran through different drivers must not be fused into one curve.

    Each bucket is a separately bootable call, so a replaced container can land on a host carrying
    a different NVIDIA driver mid-rollout. Same weights, same card model, different execution
    stack. The per-bucket probes captured the distinction and the summary discarded it, keeping
    only the canary's runtime packages and combining the curves regardless.
    """
    namespace = _bench_namespace("_require_one_identity", "_dispatch_stack_identity")
    gate = namespace["_require_one_identity"]
    dispatch_identity = namespace["_dispatch_stack_identity"]

    def require(payloads: list[tuple[str, dict[str, Any]]]) -> dict[str, str]:
        return gate(payloads, dispatch_identity, "kernel-dispatch stack")

    same = [
        ("canary", {"gpu": {"driver_version": "580.65.06"}}),
        ("bucket 'short_interactive'", {"gpu": {"driver_version": "580.65.06"}}),
    ]
    assert require(same) == {"driver_version": "580.65.06"}

    drifted = [
        ("canary", {"gpu": {"driver_version": "580.65.06"}}),
        ("bucket 'near_32k'", {"gpu": {"driver_version": "570.86.15"}}),
    ]
    with pytest.raises(RuntimeError, match="kernel-dispatch stacks"):
        require(drifted)


def test_the_summary_names_the_gdn_backend_that_produced_its_curves() -> None:
    """A summary archived apart from its bucket files must still name its prefill kernel.

    FlashInfer and the Triton fallback are materially different speeds, and the harness names the
    GDN backend a publication gate -- `_require_resolved_gdn_backend` refuses to start paid work
    whose kernel path cannot be labelled. But the summary carried the accepted checkpoint, the
    accepted driver and the runtime packages while dropping every probe's `gdn_prefill`, so the one
    artifact that presents the envelope could not say which kernel produced it. Each bucket is a
    separately bootable call, so the backend is also an axis two buckets can genuinely disagree on.
    """
    namespace = _bench_namespace("_require_one_identity", "_gdn_backend_identity")
    gate = namespace["_require_one_identity"]
    gdn_identity = namespace["_gdn_backend_identity"]

    def require(payloads: list[tuple[str, dict[str, Any]]]) -> dict[str, str]:
        return gate(payloads, gdn_identity, "GDN prefill backend")

    same = [
        ("canary", {"gdn_prefill": {"resolved": "flashinfer"}}),
        ("bucket 'short_interactive'", {"gdn_prefill": {"resolved": "flashinfer"}}),
    ]
    assert require(same) == {"resolved": "flashinfer"}

    drifted = [
        ("canary", {"gdn_prefill": {"resolved": "flashinfer"}}),
        ("bucket 'near_32k'", {"gdn_prefill": {"resolved": "triton"}}),
    ]
    with pytest.raises(RuntimeError, match="GDN prefill backends"):
        require(drifted)

    # The accepted backend must reach the artifact, not merely be compared and discarded: the
    # finding is about what a detached summary can identify, so the gate alone does not close it.
    source = BENCH_APP.read_text(encoding="utf-8")
    sweep_body = source[source.index("def _run_sweep_lane(") :]
    assert "accepted_gdn_backend = _require_one_identity(" in sweep_body, (
        "the sweep never establishes one GDN backend across its buckets"
    )
    summary = sweep_body[sweep_body.index('f"summary-') - 2500 : sweep_body.index('f"summary-')]
    assert '"accepted_gdn_backend": accepted_gdn_backend' in summary, (
        "the summary payload omits the GDN prefill backend the sweep accepted"
    )


def test_the_summary_refuses_buckets_that_measured_different_kv_pools() -> None:
    """Concurrency at a long context is bounded by the KV pool, not by `max_num_seqs` alone.

    `near_32k` is the case that makes this load-bearing: its whole claim is how many 32k requests
    fit at once, and that IS the block count. A replacement container sized to a different pool --
    a host with less free VRAM, a different fraction actually granted -- produces a genuinely
    different ceiling, and each bucket gates ITSELF and passes. Without this axis the two curves
    fuse into one envelope describing no engine that exists.
    """
    namespace = _bench_namespace("_require_one_identity", "_kv_pool_identity")
    gate = namespace["_require_one_identity"]
    kv_identity = namespace["_kv_pool_identity"]

    def require(payloads: list[tuple[str, dict[str, Any]]]) -> dict[str, str]:
        return gate(payloads, kv_identity, "KV pool")

    same = [
        ("canary", {"kv_cache": {"num_gpu_blocks": 12040, "block_size": 16}}),
        ("bucket 'near_32k'", {"kv_cache": {"num_gpu_blocks": 12040, "block_size": 16}}),
    ]
    assert require(same) == {"num_gpu_blocks": "12040", "block_size": "16"}

    drifted = [
        ("canary", {"kv_cache": {"num_gpu_blocks": 12040, "block_size": 16}}),
        ("bucket 'near_32k'", {"kv_cache": {"num_gpu_blocks": 8032, "block_size": 16}}),
    ]
    with pytest.raises(RuntimeError, match="KV pools"):
        require(drifted)

    # An unreadable pool must REFUSE, never resolve to a comparable stand-in. `probe_engine_kv_cache`
    # omits each field independently, so a build can expose `block_size` while withholding the block
    # count. Reporting the missing axis as `"unknown"` made it comparable, and two probes that each
    # withheld it then compared EQUAL -- publishing `accepted_kv_pool: unknown` as a verified
    # identity over buckets that could have been sized differently. Each axis is pinned separately
    # because a single both-absent case is satisfied by whichever axis raises first.
    for absent in ("num_gpu_blocks", "block_size"):
        full = {"num_gpu_blocks": 12040, "block_size": 16}
        partial = {"kv_cache": {k: v for k, v in full.items() if k != absent}}
        with pytest.raises(RuntimeError, match="no KV pool"):
            kv_identity(partial)
        with pytest.raises(RuntimeError, match="no KV pool"):
            require([("canary", {"kv_cache": full}), ("bucket 'near_32k'", partial)])

    # Two probes that both failed to expose a pool agree on NOTHING. This is the case the
    # `"unknown"` spelling silently PASSED: identical placeholders on both sides compare equal, so
    # the gate certified a pool nobody ever read. Both orderings, and the both-absent pair.
    with pytest.raises(RuntimeError, match="no KV pool"):
        require(
            [
                ("canary", {"kv_cache": {"num_gpu_blocks": 12040, "block_size": 16}}),
                ("bucket 'near_32k'", {}),
            ]
        )
    with pytest.raises(RuntimeError, match="no KV pool"):
        require([("canary", {}), ("bucket 'near_32k'", {"kv_cache": {}})])

    # The gate must run in the sweep and reach the artifact, not merely exist as a helper.
    source = BENCH_APP.read_text(encoding="utf-8")
    sweep_body = source[source.index("def _run_sweep_lane(") :]
    assert "accepted_kv_pool = _require_one_identity(" in sweep_body, (
        "the sweep never establishes one KV pool across its buckets"
    )
    summary = sweep_body[sweep_body.index('f"summary-') - 2500 : sweep_body.index('f"summary-')]
    assert '"accepted_kv_pool": accepted_kv_pool' in summary, (
        "the summary payload omits the KV pool the sweep accepted"
    )


def test_the_serving_sources_the_image_uploads_are_recorded_in_every_artifact() -> None:
    """An edit to the serving engine must move recorded provenance.

    The image does ``.add_local_python_source("flash")`` -- the WHOLE package -- but until this
    guard nothing in an artifact covered it. ``workload_checksum()`` digests a named list of bench
    functions and reads only from ``flash.serving.bench``; ``_serving_patch_identity()`` digests the
    vLLM patch alone. So editing ``flash/serving/src/engine/generation.py`` or ``lora_engine.py``
    changed what every measured request executed while the patch digest, the workload checksum, the
    resolved package versions and the checkpoint commits all stayed byte-identical. Two campaigns
    running materially different serving code would compare as compatible.

    The digest must be over CONTENT, not a version string: a hand-maintained constant moves when
    someone remembers to move it, which is the same defect one level up.
    """
    source = BENCH_APP.read_text(encoding="utf-8")

    # Every provenance block that records the patch must also record the sources, or an artifact
    # from the un-covered path is exactly as unfalsifiable as before.
    assert source.count('"serving_patch": _serving_patch_identity()') == source.count(
        '"serving_source": _serving_source_identity()'
    ), "a provenance block records the vLLM patch but not the flash sources shipped beside it"

    namespace = _bench_namespace(
        "_serving_source_identity",
        REPO_DIR=REPO_ROOT,
        hashlib=hashlib,
    )
    identity = namespace["_serving_source_identity"]()
    assert identity["package"] == "flash"
    assert identity["file_count"] > 100, (
        "the digest covers a handful of files; it is not reading the uploaded package"
    )
    assert len(identity["sha256"]) == 64

    # The property that makes it evidence: editing a serving file the digest is supposed to cover
    # must change the value. Done against a COPY of the tree so the real checkout is untouched.
    # No `dir=` here: every other temp directory in this file honours TMPDIR, and pinning an
    # absolute path made the guard pass locally and fail on CI, where /mnt/resource does not exist.
    with tempfile.TemporaryDirectory() as tmp:
        mirror = Path(tmp) / "repo"
        (mirror / "flash").mkdir(parents=True)
        shutil.copytree(REPO_ROOT / "flash", mirror / "flash", dirs_exist_ok=True)
        mirrored = _bench_namespace("_serving_source_identity", REPO_DIR=mirror, hashlib=hashlib)[
            "_serving_source_identity"
        ]
        before = mirrored()
        assert before["sha256"] == identity["sha256"], (
            "an identical copy of the tree digests differently; the digest is not content-addressed"
        )

        engine_file = mirror / "flash" / "serving" / "src" / "engine" / "generation.py"
        assert engine_file.exists(), "the serving generation module moved; repoint this guard"
        engine_file.write_bytes(engine_file.read_bytes() + b"\n# edited\n")
        assert mirrored()["sha256"] != before["sha256"], (
            "editing the serving generation path left the recorded source identity unchanged"
        )

        # A rename with identical bytes must move it too: the module path decides what an import
        # resolves to, so a moved file changes execution even when nothing inside it changed.
        engine_file.write_bytes(engine_file.read_bytes().replace(b"\n# edited\n", b""))
        assert mirrored()["sha256"] == before["sha256"]
        renamed = engine_file.with_name("generation_moved.py")
        engine_file.rename(renamed)
        assert mirrored()["sha256"] != before["sha256"], (
            "renaming a serving module left the recorded source identity unchanged"
        )


def test_the_probe_records_the_driver_the_kernels_actually_dispatch_through() -> None:
    """`torch.version.cuda` is a build-time constant of the wheel, not the host's driver.

    It reports the CUDA toolkit torch was COMPILED against, so it is identical on every host running
    the same pinned image. What dispatches kernels -- and what moves the performance of an unchanged
    image -- is the host NVIDIA driver. Without it, two blocks measured months apart on the same
    image can differ materially while every recorded version string stays byte-identical, and the
    difference is unexplainable from the artifact.
    """
    import flash.serving.bench.probe as probe_module

    calls: list[str] = []

    class _FakeNvml:
        NVML_SUCCESS = 0

        def nvmlInit(self) -> None:
            calls.append("init")

        def nvmlShutdown(self) -> None:
            calls.append("shutdown")

        def nvmlDeviceGetCount(self) -> int:
            return 1

        def nvmlDeviceGetHandleByIndex(self, index: int) -> object:
            return object()

        # Returned as BYTES, the way pynvml's C-binding build does. The driver read must decode it
        # the same way the device name does, or the artifact carries a `b'580.65.06'` repr.
        def nvmlDeviceGetName(self, handle: object) -> bytes:
            return b"NVIDIA B200"

        def nvmlDeviceGetCudaComputeCapability(self, handle: object) -> tuple[int, int]:
            return (10, 0)

        def nvmlDeviceGetMemoryInfo(self, handle: object) -> object:
            return types.SimpleNamespace(total=183_000_000_000)

        def nvmlSystemGetDriverVersion(self) -> bytes:
            calls.append("driver")
            return b"580.65.06"

    with mock.patch.dict(sys.modules, {"pynvml": _FakeNvml()}):
        result = probe_module.probe_gpu()

    assert result["available"] is True
    assert "driver" in calls, "the probe never asked NVML for the host driver version"
    assert result.get("driver_version") == "580.65.06", (
        "the driver version is missing or undecoded; an artifact cannot explain a driver-driven "
        "performance change without it"
    )
    # It must be a SEPARATE field, not a replacement: the toolkit torch was built against still
    # explains an ABI mismatch, and conflating the two loses which one moved.
    assert result["driver_version"] != result.get("cuda_version")
