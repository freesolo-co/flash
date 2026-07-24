# pure, gpu-free decision logic for the openrlhf opd rollout vllm engine runtime.
#
# the openrlhf opd child colocated-vllm rollout engine is created by openrlhf's
# create_vllm_engines -> RolloutRayActor.__init__ -> vllm.AsyncEngineArgs(**kwargs). openrlhf's
# --vllm.* cli surface only exposes num_engines/tp/gpu_memory_utilization/enable_prefix_caching/
# enforce_eager/enable_sleep/sync_backend, so fp8 kv cache, chunked prefill, dynamic rollout
# concurrency, and the gpu-family compile/backend safeguards that the trl opd worker
# (flash/engine/worker/opd_vllm.py) applies are missing on the openrlhf path.
#
# this module holds the *decision* half only (no torch, no vllm, no side effects): given the live
# compute capability, vllm version, model id, and sizing inputs it returns the extra AsyncEngineArgs
# kwargs to inject, the process env to set, and a metadata record. the child sitecustomize computes
# the live inputs in the rollout actor (where the gpu is visible) and applies the result. keeping the
# decision pure makes the fp8 gate, the enforce_eager family gate, and the sizing floor unit-testable
# off-gpu, matching the trl contract without a live rollout.

from __future__ import annotations

# fp8 kv cache is an ada/hopper/blackwell feature; enable exactly when compute capability >= (8, 9),
# the same threshold as flash.providers.base.supports_fp8_kv (_FP8_KV_MIN_CAPABILITY).
_FP8_KV_MIN_CAPABILITY = (8, 9)

# validated rollout-graph families: a100 (8, 0) and h100 (9, 0) keep vllm cuda graphs. every other
# family on vllm >= 0.19 hit the 0.19.x aot_compile/slot-mapping crash and must fall back to eager,
# except blackwell (major 10/12) which uses decode-only cuda graphs with torch.compile disabled.
_GRAPH_VALIDATED_CC = {(8, 0), (9, 0)}
_BLACKWELL_MAJORS = (10, 12)


def _as_cc(cc) -> tuple[int, int] | None:
    """coerce a (major, minor) compute-capability pair; None if unusable."""
    try:
        major, minor = int(cc[0]), int(cc[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (major, minor)


def _as_version(vllm_version) -> tuple[int, int, int]:
    """(major, minor, patch) from a vllm version string like '0.22.1+cu129'; (0,0,0) if unparseable."""
    try:
        base = str(vllm_version).split("+", 1)[0]
        parts = [int(x) for x in base.split(".")[:3]]
        while len(parts) < 3:
            parts.append(0)
        return (parts[0], parts[1], parts[2])
    except (TypeError, ValueError):
        return (0, 0, 0)


def opd_rollout_runtime_plan(
    *,
    cc,
    vllm_version,
    seq_cap: int,
    max_num_seqs: int | None,
    card_gb: float | None,
    mamba_block_size: int = 0,
) -> dict:
    """decide the extra opd rollout vllm engine runtime, gpu-free.

    ``cc`` is the live (major, minor) compute capability, ``vllm_version`` the live vllm version
    string, ``max_num_seqs`` the parent-sized rollout concurrency (dynamic sizing happens parent-side
    from the model/gpu; this module only carries it onto the engine), ``card_gb`` the device total
    memory, and ``mamba_block_size`` the model's mamba block size (0 for non-mamba models; passed in
    because the child cannot import the flash catalog). the mamba scheduler-budget floor mirrors
    ``flash.catalog.opd_mamba_batched_token_floor``: on sub-140GiB cards, when the derived vllm budget
    ``max_num_seqs * seq_cap`` is below ``mamba_block_size`` we pin ``max_num_batched_tokens`` to the
    block size so the mamba/gdn scheduler has a valid budget.

    Returns ``{"kwargs": {...}, "env": {...}, "meta": {...}}``:
      * ``kwargs`` -> extra ``vllm.AsyncEngineArgs`` kwargs to merge into the rollout engine.
      * ``env``    -> process env the actor must set before engine construction.
      * ``meta``   -> a json-safe record of the applied decisions for train metadata/telemetry.
    """
    cc_pair = _as_cc(cc)
    version = _as_version(vllm_version)
    kwargs: dict = {}
    env: dict = {}
    meta: dict = {
        "cc": list(cc_pair) if cc_pair else None,
        "vllm_version": ".".join(str(x) for x in version),
    }

    # --- fp8 kv cache (cc >= 8.9) ---
    fp8_kv = bool(cc_pair is not None and cc_pair >= _FP8_KV_MIN_CAPABILITY)
    if fp8_kv:
        kwargs["kv_cache_dtype"] = "fp8"
    meta["kv_cache_dtype"] = kwargs.get("kv_cache_dtype")

    # --- chunked prefill (always on, matching the trl OPDVllmConfig default) ---
    kwargs["enable_chunked_prefill"] = True
    meta["enable_chunked_prefill"] = True

    # --- dynamic rollout concurrency + mamba batched-token floor ---
    if max_num_seqs is not None and int(max_num_seqs) > 0:
        kwargs["max_num_seqs"] = int(max_num_seqs)
    meta["max_num_seqs"] = kwargs.get("max_num_seqs")
    if (
        card_gb is not None
        and float(card_gb) < 140
        and int(mamba_block_size) > 0
        and kwargs.get("max_num_seqs")
    ):
        derived_budget = int(kwargs["max_num_seqs"]) * int(seq_cap)
        if derived_budget < int(mamba_block_size):
            kwargs["max_num_batched_tokens"] = int(mamba_block_size)
    meta["max_num_batched_tokens"] = kwargs.get("max_num_batched_tokens")

    # --- gpu-family compile/backend safeguards + enforce_eager gate ---
    # validated a100/h100 keep vllm cuda graphs (enforce_eager stays openrlhf's default False). every
    # other family on vllm >= 0.19 falls back to eager to dodge the 0.19.x aot_compile/slot-mapping
    # crash. blackwell additionally forces the in-process v1 engine core to avoid the SyncMPClient
    # startup stall. this restores the trl opd safeguards; the blackwell decode-only cuda-graph
    # *optimization* (enforce_eager=False + a compilation_config) needs a non-serializable vllm object
    # and a live blackwell rollout to validate, so it is intentionally deferred as a follow-up.
    enforce_eager = None
    reason = None
    if cc_pair is not None and version >= (0, 19, 0):
        if cc_pair in _GRAPH_VALIDATED_CC:
            reason = f"cc={cc_pair[0]}.{cc_pair[1]} validated (a100/h100): keep vllm cuda graphs"
        else:
            enforce_eager = True
            kwargs["enforce_eager"] = True
            reason = (
                f"cc={cc_pair[0]}.{cc_pair[1]} unvalidated family on vllm>=0.19: enforce_eager to "
                "avoid 0.19.x aot_compile/slot-mapping crash"
            )
            if cc_pair[0] in _BLACKWELL_MAJORS:
                env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
                reason += "; blackwell in-process v1 engine core to avoid SyncMPClient startup stall"
    meta["enforce_eager"] = enforce_eager
    meta["enforce_eager_reason"] = reason
    meta["fp8_kv"] = fp8_kv

    return {"kwargs": kwargs, "env": env, "meta": meta}


def install_rollout_runtime_shim(
    rollout_actor_cls,
    plan_provider,
    *,
    meta_writer=None,
    env_setter=None,
):
    """wrap an openrlhf RolloutRayActor's runtime ``__init__`` to inject the opd vLLM runtime plan.

    openrlhf's ``create_vllm_engines`` builds the rollout engine kwargs internally with no seam for
    extra ``AsyncEngineArgs`` fields, so the injection happens by wrapping the actor's async
    ``__init__`` (which funnels its ``**kwargs`` into ``vllm.AsyncEngineArgs``). the ray-generated
    ``modified_class`` is the class ray actually instantiates, so the patch lands there when present.

    ``plan_provider`` is a zero-arg callable returning an :func:`opd_rollout_runtime_plan` result. the
    child passes one that reads the live gpu inside the actor; tests pass a fixed plan so the wiring
    is verifiable off-gpu. ``meta_writer`` (optional) records ``plan["meta"]`` (sidecar for
    train_meta). ``env_setter`` (optional) applies ``plan["env"]`` (defaults to ``os.environ``).
    the wrapper never raises out of plan construction: a shim failure must not break rollout startup.
    returns the patched runtime class.
    """
    import functools
    import os

    ray_metadata = getattr(rollout_actor_cls, "__ray_metadata__", None)
    runtime_cls = getattr(ray_metadata, "modified_class", None) or rollout_actor_cls
    original_init = runtime_cls.__init__

    def _apply_env(key, value):
        if env_setter is not None:
            env_setter(key, value)
        else:
            os.environ[str(key)] = str(value)

    @functools.wraps(original_init)
    async def _wrapped_init(self, *args, **kwargs):
        try:
            plan = plan_provider()
            for key, value in plan.get("env", {}).items():
                _apply_env(key, value)
            kwargs.update(plan.get("kwargs", {}))
            if meta_writer is not None:
                meta_writer(plan.get("meta", {}))
            print("[opd] vLLM rollout runtime plan applied: " + str(plan.get("meta", {})))
        except Exception as exc:  # a shim failure must never break rollout startup
            print("[opd][warn] vLLM runtime shim failed; using openrlhf defaults: " + str(exc))
        return await original_init(self, *args, **kwargs)

    runtime_cls.__init__ = _wrapped_init
    return runtime_cls
