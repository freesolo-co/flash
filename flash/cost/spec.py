"""Map a parsed training ``JobSpec`` to a cost ``RunConfig`` / step count / estimate.

Shared by the CLI (``slm train --cost``) and the control plane (which charges the estimate
to the user's org at submit), so both price the SAME work the run will bill for.

The CHARGE-vs-QUOTE invariant (PR #3 review): ``slm train --cost`` resolves the spec fully
OFFLINE (the CLI scopes ``FLASH_SKIP_NET`` over parse + sizing), but the control plane parses
the submitted spec WITHOUT that guard — so for a ``model_policy = "allow"`` unlisted model,
the server's parse can read the real param count from the HF API and resolve a different
(e.g. larger) GPU class into ``spec.gpu.type`` than the offline CLI quote showed.
``offline_estimate_for_spec`` re-prices the spec on the SAME offline basis the user saw
(``FLASH_SKIP_NET`` forced over the estimate, and a policy-word GPU re-resolved offline from
``gpu.requested``), so the amount charged equals the amount quoted for the same spec.
"""

from __future__ import annotations

from flash.cost.analytical import estimate_cost
from flash.cost.types import CostEstimate, RunConfig


def sft_realized_batch(batch_size: int) -> int:
    """The SFT global batch the WORKER realizes for a requested ``batch_size``.

    The worker (engine.worker) never trains at the raw requested batch: it fixes the per-device
    micro-batch at ``_sft_per_device_bs()`` (4) and reaches the target via CEIL grad-accum, so the
    realized global batch is ``per_device x ceil(target / per_device)`` -- which can EXCEED the
    request when it isn't a multiple of the micro-batch (e.g. 16/6 -> per_device 4, accum 2 -> 8).
    Mirror that here so the optimizer-step count matches what actually runs.
    """
    from flash.engine.vram import _sft_per_device_bs

    target = max(1, int(batch_size))
    per_device = max(1, min(_sft_per_device_bs(), target))
    grad_accum = max(1, -(-target // per_device))  # ceil
    return per_device * grad_accum


def count_env_examples(env_id: str, params: dict | None = None) -> int | None:
    """Number of training rows in ``env_id``'s dataset, or ``None`` if it can't be loaded.

    Loads the verifiers env (installed via ``slm env install``) and counts its train split --
    the same ``ACTIVE_ENV.dataset("train")`` the worker trains over -- so an SFT run with no
    ``[train].max_examples`` cap is priced on the REAL dataset size, not a guess. Best-effort:
    returns ``None`` on any failure (env not installed, missing deps, load/download error) so
    the caller can surface a clear message rather than bill a wrong number.
    """
    if not env_id:
        return None
    try:
        from flash.envs import load_environment

        rows = load_environment(env_id, params or {}).dataset("train")
    except Exception:
        return None
    return len(rows) if rows is not None else None


def spec_steps(spec) -> int:
    """Per-seed optimizer steps implied by a train spec (mirrors the worker).

    GRPO carries ``train.steps`` (default recipe ``num_steps``) -- ``train.steps`` is a GRPO
    concept and is NOT consulted for SFT. SFT runs by epochs over a (capped) dataset, so steps =
    ``epochs x ceil(num_examples / realized_batch)``, capped by ``max_steps``, where ``epochs``
    defaults to ``RECIPE.sft.num_epochs`` (2), ``realized_batch`` is the worker's grad-accum global
    batch (``sft_realized_batch``), and ``num_examples`` is ``max_examples`` if pinned else the
    REAL env train-split size (``count_env_examples``) -- the full dataset the worker trains over.
    """
    from flash.engine.recipe import RECIPE

    t = spec.train
    if spec.algorithm == "grpo":
        if t.steps is not None:
            return max(1, int(t.steps))
        return RECIPE.rl.num_steps
    # --- SFT ---
    cap = int(t.max_steps) if t.max_steps else 0  # SFT-only optimizer-step cap (0 = uncapped)
    epochs = int(t.epochs) if t.epochs is not None else RECIPE.sft.num_epochs
    requested_batch = int(t.batch_size) if t.batch_size is not None else RECIPE.sft.effective_batch
    batch = sft_realized_batch(requested_batch)  # worker's grad-accum-realized global batch
    if t.max_examples is not None:
        examples = int(t.max_examples)
    else:
        # No cap: the worker trains the FULL env dataset, so price its real size.
        examples = count_env_examples(spec.environment.id, spec.environment.params)
        if examples is None:
            raise ValueError(
                f"could not load environment {spec.environment.id!r} to count its training "
                f"examples for the cost; install it (`slm env install {spec.environment.id}`) "
                "or pin [train].max_examples"
            )
    n = max(1, -(-examples // batch) * epochs)  # epochs x ceil(examples / realized_batch)
    return min(n, cap) if cap > 0 else n


def runconfig_from_spec(spec, *, offline_gpu: bool = False) -> RunConfig:
    """Map a parsed training ``JobSpec`` to a cost ``RunConfig``.

    Each seed is its own job that re-pays the cold start (runner.py), so scale both the step
    count and the setup repeats by the seed count -- the estimate then prices the same total
    work the run would bill for.

    ``offline_gpu``: when the GPU was a policy word (``auto``/``cheapest``), forward that word
    (from ``gpu.requested``) instead of the parse-time resolved ``gpu.type``, so the estimator
    re-picks the class from its own OFFLINE VRAM sizing. This makes the charge match the
    offline ``--estimate`` quote even when the server resolved ``gpu.type`` from a live HF probe
    (a concrete user pin is always honored as-is). See ``offline_estimate_for_spec``.
    """
    t, g = spec.train, spec.gpu
    is_grpo = spec.algorithm == "grpo"
    seeds = max(1, len(t.seeds or (0,)))
    gpu = g.type
    if offline_gpu and _requested_is_policy(g):
        # Re-resolve the cheapest fitting class from the estimator's offline sizing rather
        # than trusting the (possibly HF-sized) parse-time pin.
        gpu = (g.requested or "auto").strip().lower()
    return RunConfig(
        model_id=spec.model,
        method=spec.algorithm,
        steps=spec_steps(spec) * seeds,
        setup_repeats=seeds,
        seq_len=t.max_length,
        completion_len=t.max_tokens if is_grpo else None,
        batch_size=t.batch_size,
        group_size=t.group_size if is_grpo else None,
        lora_rank=t.lora_rank,
        thinking=spec.thinking,
        gpu=gpu,
        provider=g.provider,
        allow_unvalidated=g.allow_unvalidated,
        max_wall_seconds=g.max_wall_seconds,
        environment=spec.environment.id or None,
    )


def _requested_is_policy(gpu_spec) -> bool:
    """True when the user asked for a policy word (``auto``/``cheapest``), not a concrete pin."""
    from flash.providers.base import POLICY_NAMES

    return (gpu_spec.requested or "").strip().lower() in POLICY_NAMES


def estimate_for_spec(spec) -> CostEstimate:
    """The pre-flight ``CostEstimate`` for a parsed training ``JobSpec`` (as parsed)."""
    return estimate_cost(runconfig_from_spec(spec))


def offline_estimate_for_spec(spec) -> CostEstimate:
    """The OFFLINE-consistent estimate -- what ``slm train --cost`` would have quoted.

    Reproduces the CLI's offline quote WITHOUT mutating the process-wide ``FLASH_SKIP_NET``
    env: ``estimate_cost`` already forces offline HF sizing (``select_gpu`` passes
    ``skip_net=True`` to ``required_vram_gb``), and ``offline_gpu=True`` forwards a policy-word
    GPU so the class is re-picked from that offline VRAM rather than the (possibly HF-sized)
    parse-time pin. Avoiding the env flip is important on the control plane, where concurrent
    requests (auth verify, provider pricing, other submits) must not observe a flipped global
    and unexpectedly skip their own network I/O (PR #3 review). The control plane charges THIS
    amount, so the org is billed exactly what the user was quoted (and, paired with the
    ``min(...)`` guard in ``charge_run_estimate``, never more than that quote).
    """
    return estimate_cost(runconfig_from_spec(spec, offline_gpu=True))
