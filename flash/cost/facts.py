"""Catalog facts and provider-aware pricing accessors for the cost model."""

from __future__ import annotations

from flash.catalog import MODELS
from flash.providers.base import GPU_INFO, GpuClass, providers_for

GPU_COMPUTE_TFLOPS: dict[str, float] = {
    # A10: 125 TFLOPS dense bf16 tensor (NVIDIA spec); Lambda-only 24 GB class, else defaults to 100.
    "A10": 125.0,
    # MEASURED on a rented RunPod RTX 4090 (sm89, torch 2.10+cu128): 162.2-162.7 TFLOPS sustained
    # on large square bf16 matmuls, against the 165 vendor spec -- within 1.5%, so the spec number
    # is a fair proxy for this class and is kept.
    "RTX 4090": 165.0,
    "RTX 5090": 210.0,
    # MEASURED on a rented RunPod A100 80GB PCIe (sm80, torch 2.10+cu128): 242-259 TFLOPS sustained
    # across repeated trials at full clocks (1410 MHz) and 31C, i.e. neither thermal- nor
    # power-limited. The 312 figure is the vendor dense-bf16 spec, which real matmul kernels do not
    # reach; using it overstated A100 throughput by ~20% and made the class look faster (and so
    # cheaper per job) than it is. Set to the measured sustained rate.
    "A100 PCIe": 250.0,
    # MEASURED on a rented RunPod A100-SXM4-80GB (sm80, torch 2.10+cu128): 263-265 TFLOPS across
    # repeated trials. Same 108 SMs as the PCIe part but a 400 W envelope against 300 W, which is
    # what the ~6% gain over the measured PCIe rate reflects. Both are far below the shared 312
    # vendor spec.
    "A100 SXM": 264.0,
    # A100 SXM 40GB: same SMs/tensor cores as the 80GB A100 SXM, less HBM only, so it inherits the
    # measured SXM rate. Without an entry, 33-40 GB Lambda/Vast quotes fall back to _DEFAULT_TFLOPS.
    "A100 SXM 40GB": 264.0,
    # MEASURED on a rented RunPod H100 PCIe (sm90, torch 2.10+cu128): 490-504 TFLOPS sustained
    # across repeated trials at full clocks (1755 MHz), 310 W board. The 990 figure is the SXM
    # part's spec; the PCIe board this class actually provisions delivers about half of it. Using
    # 990 made H100 look twice as fast as it is and let it win nearly every ranking decision on
    # throughput it cannot deliver. Set to the measured sustained rate.
    "H100": 500.0,
    # H200: same SM/tensor-core configuration as H100 with more HBM and a higher power envelope.
    # Scaled from the H100 measurement rather than the SXM spec. Not directly measured.
    "H200": 550.0,
    "RTX Pro 6000": 250.0,
    # B200: 2.25 PFLOPS bf16 dense (NVIDIA spec); prevents ~10x cost over-estimate vs _DEFAULT_TFLOPS.
    "B200": 2250.0,
}
_DEFAULT_TFLOPS = 100.0

# classes whose cards talk to each other over nvlink rather than the pcie root complex. this is the
# single largest input to multi-card scaling: the same 2-card benchmark measured 1.7675x on an
# nvlink pair and 1.4212x on a pcie pair, so one global scaling constant cannot describe both.
# membership is by form factor -- sxm datacenter parts carry nvlink, pcie boards and every geforce
# part do not (the 4090 dropped nvlink entirely). anything absent is treated as pcie, which is the
# conservative side: it under-credits a combination rather than ranking it on bandwidth it lacks.
#
# classify by the board a MULTI-CARD run actually lands on, which is runpod's pin -- not the
# cheapest board the class can serve a single-gpu run on. vast also searches multi-card now (it
# iterates rentable_gpu_counts with num_gpus=count), and it sells the class as a market rather than
# a pinned board, so membership here is NOT sufficient for a vast combination: see
# _PCIE_AMBIGUOUS_PROVIDERS below. lambda still has no gpu_count path.
_NVLINK_CLASSES: frozenset[str] = frozenset(
    {
        # MEASURED: 2x A100-SXM4-80GB on RunPod reached 1.7675x (see MULTI_CARD_SCALING_NVLINK).
        "A100 SXM",
        # same sxm4 board and nvlink fabric as the 80gb part, less hbm only.
        "A100 SXM 40GB",
        # runpod pins H100 to NVIDIA_H100_80GB_HBM3 (providers/base.py) -- the sxm part, with
        # nvlink. the pcie and NVL boards in runpod's ADA_80_PRO pool are explicitly NEGATED by
        # providers/runpod/gpus.py, so a multi-card H100 combination lands on sxm silicon. the
        # measured-tflops comment on the H100 entry above refers to a single-gpu PCIe rental and
        # says nothing about which board a multi-card run gets.
        "H100",
        # sxm parts with nvlink/nvswitch by form factor. INFERRED, not measured -- no multi-card
        # benchmark has been run on either class.
        "H200",
        "B200",
    }
)


# providers that sell a canonical class as a market rather than a pinned board, so an nvlink
# class membership cannot be trusted for a multi-card combination there. vast lists pcie parts under
# aliases that normalize into nvlink-classed entries and its offers carry no interconnect field.
# runpod is absent because it pins an exact gpu id per class; lambda is absent because it has no
# multi-card path at all, so no scaling question arises.
_PCIE_AMBIGUOUS_PROVIDERS: frozenset[str] = frozenset({"vast"})


def gpu_tflops(name: str) -> float:
    """Peak bf16 tensor TFLOPS for a managed GPU class."""
    return GPU_COMPUTE_TFLOPS.get(name, _DEFAULT_TFLOPS)


def has_nvlink(name: str, provider: str = "") -> bool:
    """Whether cards of class ``name`` are interconnected by nvlink rather than pcie.

    Unknown classes report False: a class nobody has classified is far more likely to be a pcie
    board than an sxm one, and guessing wrong in that direction only under-credits scaling.

    ``provider`` narrows the same way. Membership in ``_NVLINK_CLASSES`` is justified by RunPod's
    pin, which names one board; vast sells a whole market under the same canonical class, including
    the explicit pcie aliases (``H100 PCIE`` -> H100, ``A100 PCIE`` -> A100 SXM 40GB). A vast
    multi-card combination can therefore land on pcie boards while the class says nvlink, and the
    offer carries no topology to distinguish them. Report pcie for those, which under-credits a
    genuine sxm host rather than pricing a pcie one on bandwidth it does not have -- the direction
    the scaling constants themselves are chosen for.
    """
    if name not in _NVLINK_CLASSES:
        return False
    return provider.strip().lower() not in _PCIE_AMBIGUOUS_PROVIDERS


# realized TRAINING throughput sits well below peak when a class's training kernels don't reach it.
# b200 (sm100) has no arch-tuned training kernels yet and falls back to the same portable paths as
# h200 (sm90), so its 2.25 pflops dense-bf16 peak does not materialize for training -- realized
# throughput is h200-class (and frequently lower for rl/grpo). cap b200 at h200 so the analytical
# cost model does not rank it as faster/cheaper than h200 on peak flops alone (it is not, and is
# often slower). this is a conservative floor: it never prices b200 above h200-equivalent training
# time, so it cannot over-charge, and it removes the "b200 looks cheapest" inversion at the source.
# refine per-workload once real b200 training samples exist. the vram/serving paths keep the true
# peak via gpu_tflops; only the training-time model uses this.
_TRAIN_TFLOPS_CAP: dict[str, float] = {
    # h200-class realized training throughput, not the 2250 dense-bf16 peak. tracks the H200 entry,
    # which is itself now anchored to a measured H100 PCIe rate rather than a vendor spec.
    "B200": 550.0,
}


def effective_train_tflops(name: str) -> float:
    """Realized bf16 TRAINING throughput for the analytical cost model.

    Equal to ``gpu_tflops`` (peak) except where a class's training kernels fall short of peak: b200
    training falls back to portable kernels, so it is capped at h200-class throughput rather than
    its raw 2.25 pflops peak."""
    peak = gpu_tflops(name)
    cap = _TRAIN_TFLOPS_CAP.get(name)
    return min(peak, cap) if cap is not None else peak


def gpu_hourly_usd(
    name: str,
    provider: str | None = None,
    max_wall_seconds: float = 0.0,
    min_vram_gb: int = 0,
    gpu_type: str = "",
) -> float:
    """Representative $/hr for a class, on ``provider`` when given.

    When ``provider`` is ``lambda`` or ``vast`` and the class is offered there, price it through that
    provider's pricing module (live with a static fallback); otherwise use the RunPod static rate.

    ``max_wall_seconds`` (>0) is threaded into the Vast live market so a duration-bound quote prices
    against offers that outlast the run, not a short-lived one filtered out at launch.

    ``min_vram_gb`` (>0) floors the Vast market search at the job's required VRAM — the SAME floor
    ``pick_gpu`` selected under — so a high-VRAM class isn't crowded off the price-sorted page and
    misquoted on the static fallback (selection/quote parity).
    """
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    p = (provider or "").strip().lower()
    if p == "lambda" and info.lambda_name:
        from flash.providers import get_provider

        return get_provider("lambda").hourly_rate(name)
    if p == "vast" and info.vast_name:
        # Vast is a live market whose rates differ materially from RunPod's static ones, so price a
        # provider="vast" quote through the Vast pricing module (live + static fallback).
        from flash.providers.vast.pricing import hourly_rate

        return hourly_rate(
            name,
            max_wall_seconds=max_wall_seconds,
            min_vram_gb=min_vram_gb,
            gpu_type=gpu_type,
        )
    return info.hourly_usd


def gpu_vram_gb(name: str) -> int:
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    return info.vram_gb


def pick_gpu(
    required_vram_gb: int,
    *,
    provider: str | None = None,
    max_wall_seconds: float = 0.0,
    cost_key=None,
) -> str:
    """Cheapest GPU class that fits ``required_vram_gb``.

    No pin; every fitting class is eligible, validated or not. NOTE this is intentionally
    gate-free: the submit-time allocator restricts to the validated pool, so the
    actually-provisioned class can be pricier than the one priced here. ``provider`` restricts
    candidates to what it can provision. ``max_wall_seconds`` (>0) prices the Vast market against
    offers that outlast the run, so a long-run quote doesn't SELECT a class on the strength of a
    short-lived offer that won't survive to launch.

    ``cost_key`` is ``(gpu_name, hourly_rate) -> comparable``: pass it to rank on what the JOB
    costs rather than what the CARD costs, so a faster class wins when it finishes soon enough to
    pay for its higher rate. It is injected rather than imported because the step model lives in
    ``analytical``, which imports this module -- building the key there keeps the dependency
    one-way. Omitted (the default) ranks on $/hr, which is correct for callers that have no run to
    price and is the honest fallback for a model outside the cost catalog.
    """

    def _selectable(g: GpuClass) -> bool:
        return provider in (None, "auto") or provider in providers_for(g.name)

    candidates = [g for g in GPU_INFO.values() if g.vram_gb >= required_vram_gb and _selectable(g)]
    if not candidates:
        raise ValueError(f"no GPU class fits >= {required_vram_gb} GB")
    # Rank by the rate on the REQUESTED provider, not the RunPod nominal. For Vast, fetch the live offer
    # map ONCE (a duration-bound query bypasses the per-call cache, so pricing per candidate would fire N
    # identical market fetches) and, when reachable, restrict to classes that actually have a rentable
    # offer under the wall cap and rank by LIVE price — else a cheaper class with no surviving offer gets
    # selected/quoted on a static rate the launch path would never rent. Static fallback when offline.
    if (provider or "").strip().lower() == "vast":
        from flash.providers.vast.pricing import live_offer_rates

        live = live_offer_rates(max_wall_seconds=max_wall_seconds, min_vram_gb=required_vram_gb)
        rentable = [g for g in candidates if g.name in live] if live else []
        if rentable:
            candidates = rentable

            def _rate(g: GpuClass) -> float:
                return live[g.name]
        else:

            def _rate(g: GpuClass) -> float:
                return g.hourly_usd
    else:

        def _rate(g: GpuClass) -> float:
            return gpu_hourly_usd(g.name, provider=provider, max_wall_seconds=max_wall_seconds)

    if cost_key is not None:
        # rank on job cost, tie-breaking on rate then the same vram/name order as the $/hr path so
        # the two bases stay comparable when a run is unpriceable in only one of them.
        best = min(
            candidates, key=lambda g: (cost_key(g.name, _rate(g)), _rate(g), g.vram_gb, g.name)
        )
    else:
        best = min(candidates, key=lambda g: (_rate(g), g.vram_gb, g.name))
    return best.name


def total_params_b(model_id: str, revision: str = "") -> float:
    """Total parameter count (billions), curated entry first and HF safetensors otherwise.

    when a revision is pinned, size the pinned commit (validated against the catalog, fail-closed)
    so setup/save cost tracks the weights the worker actually loads, not the default-revision count.

    An uncataloged model reaches here only under ``model_policy="allow"`` on a self-hosted plane,
    and every quote on that path runs through this function -- so raising for "not in the catalog"
    made an authorized open-model run unsubmittable (it failed at quoting, after parsing and
    authorizing). ``resolve_params_b`` is the single source of truth VRAM sizing already uses for
    exactly this question; using it here is what stops cost and sizing from disagreeing about how
    big a model is. Unsizeable by both raises, keeping the quote fail-closed rather than silently
    pricing an unknown model as free.
    """
    info = MODELS.get(model_id)
    if info is not None and info.params_b > 0 and not revision:
        # The managed path answers here and never touches the network.
        return info.params_b

    params_b = _sized_from_hf(model_id, revision)
    if not params_b:
        raise ValueError(
            f"could not size model {model_id!r}: it is not in the catalog "
            f"({', '.join(MODELS)}) and HuggingFace returned no safetensors parameter metadata "
            f"for it, so the run cannot be priced"
        )
    return params_b


# One quote asks "how big is this model" ~30 times (every FLOPs, memory, disk and save term, plus
# _is_moe asking twice per call). For a catalog model those are dict reads; for an open-policy model
# each one would be an HF round trip, turning a submit into ~30 sequential network calls and making
# the estimate hostage to hub latency. Memoized per process: a model id and revision name immutable
# weights, so a SUCCESSFUL answer cannot change under us.
#
# Not `functools.cache` on total_params_b: it keys on the literal argument tuple, so a caller that
# omits `revision` and one that passes "" are two entries and the model gets fetched twice. Keying
# the normalized pair here is also what lets the "only cache a hit" rule below be stated outright,
# rather than resting on the unobvious stdlib detail that `cache` does not store exceptions.
# Not on `fetch_hf_params_b` either, which tests monkeypatch -- caching there would leak a stubbed
# size across tests.
_SIZE_MEMO: dict[tuple[str, str], float] = {}


def _sized_from_hf(model_id: str, revision: str) -> float | None:
    """Resolve an uncataloged model's size, caching only a real answer.

    A miss is NOT cached. A failed lookup is a transient hub error, a rate limit, or an HF_TOKEN
    that has not been granted access yet -- not a fact about the model. Caching it would make the
    first failure permanent for the life of the process, so on a long-lived self-hosted plane one
    blip would keep rejecting that model until the operator restarted the plane, with no way to
    tell that from a genuinely unsizeable model.
    """
    key = (model_id, revision)
    if key not in _SIZE_MEMO:
        from flash.engine.vram import resolve_params_b

        params_b = resolve_params_b(model_id, revision=revision)
        if not params_b:
            return None
        _SIZE_MEMO[key] = params_b
    return _SIZE_MEMO[key]


def active_params_b(model_id: str) -> float:
    """Active params per token (billions); falls back to total for dense models. Use for FLOPs, not VRAM.

    An uncataloged model has no curated MoE routing data, so it is priced as dense (active ==
    total). That is the honest default: over-pricing a sparse model is safe, and inventing an
    active-param count for an unvalidated architecture is not.
    """
    info = MODELS.get(model_id)
    if info is None:
        return total_params_b(model_id)
    return info.active_params_b or info.params_b


def model_quant(model_id: str) -> str:
    """Quantization of the catalog entry; defaults to 'bf16'."""
    info = MODELS.get(model_id)
    return (info.quant or "bf16") if info is not None else "bf16"


def download_weight_gb(model_id: str, revision: str = "") -> float:
    """Full bf16 checkpoint size in GB (2 bytes/param)."""
    return total_params_b(model_id, revision) * 2.0


# ~1s mid-range default across grader types (regex ~0.01s to LLM judge ~3s).
AVG_REWARD_SECONDS_PER_COMPLETION = 1.0


def reward_seconds_per_completion(override: float | None = None) -> float:
    """Per-completion reward latency (s): the explicit override, else the single average."""
    if override is not None:
        return max(0.0, override)
    return AVG_REWARD_SECONDS_PER_COMPLETION


# parasail supplied-token scoring round-trip per completion.
AVG_TEACHER_SECONDS_PER_COMPLETION = 2.0


def teacher_price_per_1m(teacher_model: str) -> tuple[float, float]:
    """(input, output) $/1M tokens for a teacher model.

    Routes through the closed managed-teacher catalog. Parasail bills every supplied-token score as
    all prompt tokens plus the one generated output token required by its completions endpoint."""
    from flash.engine.recipe import resolve_teacher

    return resolve_teacher(teacher_model).usd_per_1m


def teacher_token_cost_usd(
    input_tokens: float, output_tokens: float = 0.0, teacher_model: str = ""
) -> float:
    """External teacher-API dollar cost for a token count. Deterministic; no network."""
    inp, outp = teacher_price_per_1m(teacher_model)
    return (max(0.0, input_tokens) * inp + max(0.0, output_tokens) * outp) / 1_000_000.0


def teacher_seconds_per_completion(override: float | None = None) -> float:
    """Per-completion teacher-scoring latency (s): the explicit override, else the average."""
    if override is not None:
        return max(0.0, override)
    return AVG_TEACHER_SECONDS_PER_COMPLETION
