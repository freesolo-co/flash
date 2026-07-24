# offline tests for the opd openrlhf rollout vLLM runtime plan + actor shim wiring
# (flash/engine/worker/opd_vllm_runtime.py). no torch / no vllm / no openrlhf required: the decision
# logic is pure and the shim wiring is exercised with a fake ray actor + a fixed plan provider.
import asyncio
import inspect

import pytest

from flash.engine.worker import opd_openrlhf, opd_vllm_runtime
from flash.engine.worker.opd_vllm_runtime import (
    install_rollout_runtime_shim,
    opd_rollout_runtime_plan,
)

# vllm >= 0.19 exercises the enforce_eager family gate; the worker image ships 0.22.
V = "0.22.1+cu129"


def _plan(cc, *, seq_cap=2048, max_num_seqs=8, card_gb=80.0, mamba_block_size=0, vllm=V):
    return opd_rollout_runtime_plan(
        cc=cc,
        vllm_version=vllm,
        seq_cap=seq_cap,
        max_num_seqs=max_num_seqs,
        card_gb=card_gb,
        mamba_block_size=mamba_block_size,
    )


# --- fp8 kv cache gate (cc >= 8.9) --------------------------------------------------------------
@pytest.mark.parametrize(
    ("cc", "expect_fp8"),
    [
        ((8, 0), False),  # a100 -> no fp8 kv
        ((8, 6), False),  # a10/a40
        ((8, 9), True),   # l4/l40s ada -> fp8
        ((9, 0), True),   # h100/h200
        ((10, 0), True),  # b200
        ((12, 0), True),  # rtx 5090
        (None, False),    # unknown gpu -> conservative
    ],
)
def test_fp8_kv_cache_gate(cc, expect_fp8):
    plan = _plan(cc)
    assert plan["meta"]["fp8_kv"] is expect_fp8
    assert (plan["kwargs"].get("kv_cache_dtype") == "fp8") is expect_fp8
    assert plan["meta"]["kv_cache_dtype"] == plan["kwargs"].get("kv_cache_dtype")


def test_chunked_prefill_always_enabled():
    for cc in [(8, 0), (8, 9), (9, 0), (10, 0), None]:
        plan = _plan(cc)
        assert plan["kwargs"]["enable_chunked_prefill"] is True
        assert plan["meta"]["enable_chunked_prefill"] is True


# --- dynamic rollout concurrency ----------------------------------------------------------------
def test_max_num_seqs_carried_when_positive():
    plan = _plan((9, 0), max_num_seqs=12)
    assert plan["kwargs"]["max_num_seqs"] == 12
    assert plan["meta"]["max_num_seqs"] == 12


@pytest.mark.parametrize("bad", [None, 0, -3])
def test_max_num_seqs_omitted_when_nonpositive(bad):
    plan = _plan((9, 0), max_num_seqs=bad)
    assert "max_num_seqs" not in plan["kwargs"]
    assert plan["meta"]["max_num_seqs"] is None


# --- mamba scheduler-budget floor (mirrors flash.catalog.opd_mamba_batched_token_floor) ---------
def test_mamba_floor_applied_when_budget_below_block_on_small_card():
    # derived budget = max_num_seqs * seq_cap = 2 * 1024 = 2048 < 8192 -> pin to block size.
    plan = _plan((9, 0), seq_cap=1024, max_num_seqs=2, card_gb=80.0, mamba_block_size=8192)
    assert plan["kwargs"]["max_num_batched_tokens"] == 8192
    assert plan["meta"]["max_num_batched_tokens"] == 8192


def test_mamba_floor_skipped_when_budget_already_sufficient():
    # derived budget = 8 * 2048 = 16384 >= 8192 -> no floor needed.
    plan = _plan((9, 0), seq_cap=2048, max_num_seqs=8, card_gb=80.0, mamba_block_size=8192)
    assert "max_num_batched_tokens" not in plan["kwargs"]


def test_mamba_floor_skipped_on_large_card():
    # >=140 GiB cards derive a valid budget without the floor even when the block size is large.
    plan = _plan((9, 0), seq_cap=1024, max_num_seqs=2, card_gb=141.0, mamba_block_size=8192)
    assert "max_num_batched_tokens" not in plan["kwargs"]


def test_mamba_floor_skipped_for_non_mamba_model():
    plan = _plan((9, 0), seq_cap=1024, max_num_seqs=2, card_gb=80.0, mamba_block_size=0)
    assert "max_num_batched_tokens" not in plan["kwargs"]


# --- enforce_eager family gate ------------------------------------------------------------------
@pytest.mark.parametrize("cc", [(8, 0), (9, 0)])
def test_enforce_eager_kept_off_on_validated_families(cc):
    plan = _plan(cc)
    # validated a100/h100 keep cuda graphs: shim leaves enforce_eager to openrlhf's default (False).
    assert "enforce_eager" not in plan["kwargs"]
    assert plan["meta"]["enforce_eager"] is None
    assert "VLLM_ENABLE_V1_MULTIPROCESSING" not in plan["env"]


@pytest.mark.parametrize("cc", [(8, 9), (8, 6), (7, 5)])
def test_enforce_eager_on_unvalidated_non_blackwell(cc):
    plan = _plan(cc)
    assert plan["kwargs"]["enforce_eager"] is True
    assert plan["meta"]["enforce_eager"] is True
    assert "VLLM_ENABLE_V1_MULTIPROCESSING" not in plan["env"]


@pytest.mark.parametrize("cc", [(10, 0), (12, 0)])
def test_blackwell_gets_eager_plus_inprocess_v1(cc):
    plan = _plan(cc)
    assert plan["kwargs"]["enforce_eager"] is True
    assert plan["env"]["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"


def test_old_vllm_skips_enforce_eager_gate():
    # below 0.19 the aot_compile/slot-mapping crash does not apply; do not force eager.
    plan = _plan((8, 9), vllm="0.18.0")
    assert "enforce_eager" not in plan["kwargs"]
    assert plan["meta"]["enforce_eager"] is None


def test_meta_is_json_safe_and_records_version():
    import json

    plan = _plan((9, 0))
    json.dumps(plan["meta"])  # must not raise
    assert plan["meta"]["vllm_version"] == "0.22.1"
    assert plan["meta"]["cc"] == [9, 0]


# --- shim wiring: async actor __init__ injection ------------------------------------------------
def _make_fake_actor():
    """fresh (outer, runtime) actor pair per test so patches never leak across tests.

    ``runtime`` stands in for the ray modified_class whose async __init__ funnels **kwargs into
    vllm.AsyncEngineArgs; ``outer`` exposes __ray_metadata__.modified_class like an @ray.remote class.
    """

    class _Runtime:
        async def __init__(self, *args, **kwargs):
            self.captured_args = args
            self.captured_kwargs = kwargs

    outer = type(
        "_OuterActor",
        (),
        {"__ray_metadata__": type("_Meta", (), {"modified_class": _Runtime})()},
    )
    return outer, _Runtime


def _run_init(cls, **kwargs):
    inst = cls.__new__(cls)
    asyncio.run(cls.__init__(inst, **kwargs))
    return inst


def test_shim_injects_kwargs_env_and_meta_into_modified_class():
    outer, runtime = _make_fake_actor()
    written = {}
    env_calls = {}
    fixed_plan = {
        "kwargs": {"kv_cache_dtype": "fp8", "enable_chunked_prefill": True, "max_num_seqs": 8},
        "env": {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
        "meta": {"fp8_kv": True},
    }
    install_rollout_runtime_shim(
        outer,
        lambda: fixed_plan,
        meta_writer=lambda m: written.update(m),
        env_setter=lambda k, v: env_calls.__setitem__(k, v),
    )
    inst = _run_init(runtime, model="Qwen/Qwen3.5-0.8B")
    # the plan kwargs are merged onto the engine kwargs, base kwargs preserved.
    assert inst.captured_kwargs["model"] == "Qwen/Qwen3.5-0.8B"
    assert inst.captured_kwargs["kv_cache_dtype"] == "fp8"
    assert inst.captured_kwargs["enable_chunked_prefill"] is True
    assert inst.captured_kwargs["max_num_seqs"] == 8
    assert env_calls["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert written == {"fp8_kv": True}


def test_shim_patch_lands_on_modified_class_not_outer():
    outer, runtime = _make_fake_actor()
    original_init = runtime.__init__
    install_rollout_runtime_shim(outer, lambda: {"kwargs": {}, "env": {}, "meta": {}})
    # the ray modified_class __init__ is wrapped (functools.wraps records the original as __wrapped__).
    patched = outer.__ray_metadata__.modified_class.__init__
    assert patched is not original_init
    assert getattr(patched, "__wrapped__", None) is original_init


def test_shim_falls_back_to_class_when_no_ray_metadata():
    # a plain class (no __ray_metadata__) is patched directly.
    class _Plain:
        async def __init__(self, *args, **kwargs):
            self.captured_kwargs = kwargs

    install_rollout_runtime_shim(
        _Plain, lambda: {"kwargs": {"max_num_seqs": 4}, "env": {}, "meta": {}}
    )
    inst = _run_init(_Plain, model="x")
    assert inst.captured_kwargs["max_num_seqs"] == 4


def test_shim_never_breaks_rollout_startup_on_plan_error():
    class _Boom:
        async def __init__(self, *args, **kwargs):
            self.ok = True

    def _explode():
        raise RuntimeError("gpu query failed")

    install_rollout_runtime_shim(_Boom, _explode)
    inst = _run_init(_Boom, model="x")
    # a shim failure is swallowed; the original init still runs so rollout can start.
    assert inst.ok is True


# --- embedding equivalence: the sitecustomize embeds this module verbatim ------------------------
def test_module_source_embeds_without_future_import_and_matches():
    # the child cannot import flash, so _opd_sitecustomize_extension embeds this module's source with
    # the future import stripped. re-run that exact transform, exec it standalone, and assert the
    # embedded opd_rollout_runtime_plan behaves identically to the imported one -> no drift.
    src = inspect.getsource(opd_vllm_runtime).replace("from __future__ import annotations\n", "")
    assert "from __future__ import annotations" not in src
    ns: dict = {}
    exec(compile(src, "embedded_opd_vllm_runtime.py", "exec"), ns)
    embedded = ns["opd_rollout_runtime_plan"]
    for cc in [(8, 0), (8, 9), (9, 0), (10, 0), None]:
        got = embedded(cc=cc, vllm_version=V, seq_cap=2048, max_num_seqs=8, card_gb=80.0, mamba_block_size=0)
        ref = _plan(cc)
        assert got == ref


def test_sitecustomize_source_compiles_with_embedded_module():
    # the full opd extension (embedded module + shim block + body) must be valid python.
    compile(opd_openrlhf._opd_sitecustomize_extension(), "sitecustomize.py", "exec")
    # and the embedded module + shim install symbols are present in the emitted source.
    ext = opd_openrlhf._opd_sitecustomize_extension()
    assert "def opd_rollout_runtime_plan" in ext
    assert "def install_rollout_runtime_shim" in ext
    assert "install_rollout_runtime_shim(" in ext
