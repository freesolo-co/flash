"""Head-geometry certification for rentable GPU widths.

Split out of ``flash.providers.allocator`` to keep that module under the file-size limit. The cap
is re-exported from the allocator, which is where every caller already reaches for it.
"""

from __future__ import annotations

from flash.providers.base import largest_rentable_count, rentable_gpu_counts

# Widest count safe to rent when a model's true head geometry could not be certified. Named because
# the guard that skips certification (`cap > _UNCERTIFIED_CAP`) is only correct while it matches the
# ceiling certification would otherwise apply: raise one without the other and a run either takes a
# hub round trip that cannot widen it, or skips the round trip that would have. ALLOC-004 tracks
# validating arbitrary off-catalog head geometry at every width.
_UNCERTIFIED_CAP = 4


def geometry_safe_gpu_cap(
    model_id: str, max_gpu_count: int, *, model_revision: str = "", certify: bool = False
) -> int:
    """Rentable ceiling whose head divisibility is known before paid allocation.

    The width becomes vLLM's ``tensor_model_parallel_size`` for the rollout engine (grpo
    ``train/rl/verl_config.py``, opd ``train/opd/overrides.py``), and vLLM requires
    ``num_attention_heads % tp_size == 0`` -- it raises at engine init otherwise -- so a catalog row
    is only safe at the counts that divide its OWN head count. Curated membership is not uniform
    geometry: catalog head counts are 8, 8, 16, 16, 24, and 16, so trusting membership alone
    accepted an 8-card width for the 27B (24 heads) that the engine rejects at init, after the box
    was already rented.

    This cap OUTLIVED ulysses. It was written when the width was also
    ``ulysses_sequence_parallel_size``; sequence parallelism is now pinned off on all three
    algorithms (it corrupts GatedDeltaNet state), but rollout tensor parallelism still consumes the
    same width, so the same divisibility gate is still what stands between a rented box and a
    post-payment engine failure.

    Scope, stated precisely: this certifies QUERY-head divisibility only. vLLM also constrains kv
    heads and the GDN linear dimensions under tensor parallelism, and Flash records both
    (``num_key_value_heads``, ``linear_num_value_heads``) without gating on them. Every current row
    divides 1/2/4/8 on all three axes, so nothing is mis-admitted today. Widening the check to those
    two axes is a separate invariant and deliberately not in this change.

    The head count is READ from the row (``num_attention_heads``), never derived: ``hidden_size //
    head_dim`` is a different number on four of the six rows -- see ``_query_attention_heads``.

    A revision whose geometry cannot be certified keeps the pre-existing four-card ceiling rather
    than renting 8 cards verl may reject at startup, but that ceiling only NARROWS the divisor
    search; it is not a substitute for it. A ceiling is a bound, not a divisibility proof -- 4
    divides 24 but not 20 -- so the heads are checked either way.

    A pin is not by itself unknown geometry. SFT reaches allocation with a revision ALWAYS resolved
    to a sha (``runner.submit.prepare_job`` -> ``_resolve_model_revision`` with ``required=True``),
    so treating "pinned" as "uncertifiable" capped every SFT run in the catalog at four cards and
    made ``--gpus 8`` unreachable for the algorithm that always pins -- including for a run that
    only fits at eight. The pinned commit's own ``config.json`` is what settles it: read the head
    count from that commit (validated against the catalog row, fail-closed, so a drifted pin is
    rejected rather than widened) and cap on the real number. An unreadable pin certifies nothing:
    it keeps the four-card ceiling AND falls back to the row's own head count for the divisor
    search, so it can only ever be narrower than the same run unpinned, never wider.

    ``certify`` is what permits the hub round trip, and ONLY the submission path passes it. Reading a
    pinned commit's config is network i/o, so a transient hub failure returns the uncertified
    four-card ceiling. On the submit path that is a safe conservative answer the allocator can still
    act on. On an OFFLINE path it is not: `spec_from_dict` feeds this cap to `provisional_gpu`, whose
    job is to REJECT an unplaceable run, so a blip would narrow a 35B that genuinely needs eight
    cards down to four and reject it as unplaceable during config parsing that is otherwise entirely
    offline -- turning a transient network error into a terminal, and wrong, user-facing rejection.
    The cost quote has the same shape (`_offline_gpu_shape` is documented as structural and must not
    consume live failures). Both keep the default and stay offline; certification belongs where a
    healthy retry and a real allocation decision live.
    """
    from flash.core.catalog import MODELS

    cap = largest_rentable_count(max_gpu_count)
    info = MODELS.get(model_id)
    heads = _query_attention_heads(info) if info is not None else 0
    if info is None:
        # nothing to certify a width against, and nothing to cross-check a pin's own config with.
        cap = min(cap, _UNCERTIFIED_CAP)
    elif certify and model_revision and cap > _UNCERTIFIED_CAP:
        # the weights the worker really loads are the pinned commit's, so its config -- not the
        # row's default-revision geometry -- is what may widen this run. only worth a hub round trip
        # when there is something to widen TO: at or below the uncertified cap, certification
        # cannot raise the ceiling.
        from flash.engine.plan.vram import certified_attention_heads

        certified = certified_attention_heads(model_id, model_revision)
        if certified > 0:
            heads = certified
        else:
            # uncertified: fall back to the ceiling, but keep checking the ROW's heads below. the
            # ceiling narrows the divisor search, it does not replace it, so a row whose heads do
            # not divide it is still narrowed further instead of rented at a width verl rejects.
            cap = min(cap, _UNCERTIFIED_CAP)
    if heads <= 0:
        # geometry we cannot read is geometry we cannot certify, so a catalog row that records no
        # head count is treated exactly like an uncertifiable revision rather than trusted for 8.
        return min(cap, _UNCERTIFIED_CAP)
    for count in rentable_gpu_counts(cap):
        if heads % count == 0:
            return count
    return 1


def _query_attention_heads(info) -> int:
    """Query-attention head count for a catalog row, or 0 when the row does not record one.

    Read, never derived. ``hidden_size // head_dim`` looks like the head count and is not: these
    checkpoints decouple ``head_dim`` from that ratio, so the quotient is wrong for four of the six
    catalog rows (3.5-4B is 16 heads, not 2560/256 = 10; 0.8B is 8, not 4; 3.6-27B is 24, not 20;
    35B-A3B is 16, not 8). A cap computed from the quotient divides the wrong number -- it happened
    to stay conservative on today's catalog, but nothing makes that hold for the next row added.
    """
    return int(getattr(info, "num_attention_heads", 0) or 0)
